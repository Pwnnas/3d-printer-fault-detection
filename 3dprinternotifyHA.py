"""
3D Print AI Monitor
Monitors a 3D printer via Home Assistant camera, analyzes with a local
vision model (Ollama), and sends alerts via Telegram.

SETUP:
  pip install requests python-dotenv
  Copy .env.example to .env and fill in your values.
"""

import os
import base64
import time
import threading
import socket
import logging
from datetime import datetime, timedelta

import requests
import urllib3
from dotenv import load_dotenv

# ── Load secrets from .env ─────────────────────────────────────────────────────
load_dotenv()

# 1. Home Assistant
HA_CAMERA_IMAGE_URL     = os.getenv("HA_CAMERA_IMAGE_URL")
HA_LONG_LIVED_TOKEN     = os.getenv("HA_LONG_LIVED_TOKEN")
HA_WEBHOOK_URL          = os.getenv("HA_WEBHOOK_URL")

# 2. Vision server (Ollama)
VISION_SERVER_IP        = os.getenv("VISION_SERVER_IP", "127.0.0.1")
VISION_SERVER_PORT      = int(os.getenv("VISION_SERVER_PORT", 11434))
VISION_SERVER_URL       = f"http://{VISION_SERVER_IP}:{VISION_SERVER_PORT}/v1/chat/completions"
VISION_MODEL_NAME       = os.getenv("VISION_MODEL_NAME", "llava:7b-v1.6")

# 3. Telegram
TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID        = int(os.getenv("TELEGRAM_CHAT_ID", 0))

# 4. Monitoring behaviour
POLL_INTERVAL_FAST_SECONDS   = int(os.getenv("POLL_INTERVAL_FAST_SECONDS", 20))
POLL_INTERVAL_SLOW_SECONDS   = int(os.getenv("POLL_INTERVAL_SLOW_SECONDS", 120))
FAST_PHASE_DURATION_SECONDS  = int(os.getenv("FAST_PHASE_DURATION_SECONDS", 1800))  # 30 min
EMPTY_TIMEOUT_SECONDS        = int(os.getenv("EMPTY_TIMEOUT_SECONDS", 3600))
WARN_COOLDOWN_SECONDS        = int(os.getenv("WARN_COOLDOWN_SECONDS", 600))
STOP_CONFIRM_COUNT           = int(os.getenv("STOP_CONFIRM_COUNT", 2))
WARN_ESCALATE_COUNT          = int(os.getenv("WARN_ESCALATE_COUNT", 5))
NETWORK_LISTENER_PORT        = int(os.getenv("NETWORK_LISTENER_PORT", 65432))

LOCAL_IMAGE_FILENAME    = "current_print_view.jpg"
DESCRIBE_TEMP_FILENAME  = "describe_temp_view.jpg"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("PrintMonitor")

# ── AI Prompts ─────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = """You are an expert 3D printing diagnostician. Provide a detailed, objective analysis of the 3D print shown.

Structure your response:
1. **Overall Progress & Status** - what is being printed, early/mid/late stage?
2. **Bed Adhesion** - quality at the base; any lifting, warping, or peeling?
3. **Layer Quality** - are layers clean? Any blobs (over-extrusion) or gaps (under-extrusion)?
4. **Surface Artifacts** - stringing, oozing, zits, ringing/ghosting?
5. **Summary** - one sentence overall assessment.

Constraints:
- Do NOT use the single words STOP, WARN, or OK.
- Do not suggest fixes, only describe what you see.
- Be factual and professional, like a technician reporting findings.

"""
MONITOR_PROMPT = """You are an AI 3D-print monitoring assistant. Analyze the camera image and respond with EXACTLY one of these four commands and nothing else.

STOP [reason]   - Catastrophic, unrecoverable failure (>95% certain):
                  Total detachment, severe spaghetti, major layer shift,
                  nozzle collision, extruder clearly not extruding for many layers.

WARN [reason]   - Minor defect, not yet catastrophic:
                  Minor stringing, slight warping, small blobs, support issues.

OK              - Print progressing normally.

OK EMPTY        - Printer idle, build plate clear.

Rules:
- Default to OK. Strong evidence required for WARN; near-certain disaster for STOP.
- When in doubt, respond OK.
- NEVER flag anything based on temperature, heat, or color. You cannot measure heat from a camera image.
- NEVER flag motion blur, streaks, or ghosting from a moving printhead as a problem.
- NEVER flag camera artifacts, lens glare, overexposure, or dark areas as print defects.
- A large object filling most of the frame with clean walls and good bed contact is always OK.
- Filament color (red, orange, pink, yellow) is never evidence of any problem.
- Your ENTIRE response must be one of the four formats above and nothing else.
"""
# ── Shared State ───────────────────────────────────────────────────────────────

class MonitorState:
    """Thread-safe container for shared runtime state."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latest_status: str = "Unknown (monitoring not yet run)"
        self.paused_until: datetime | None = None
        self.monitoring_started_at: float | None = None

    # ── status ────────────────────────────────────────────────────────────────
    def get_status(self) -> str:
        with self._lock:
            return self.latest_status

    def set_status(self, s: str):
        with self._lock:
            self.latest_status = s

    # ── adaptive interval ─────────────────────────────────────────────────────
    def mark_started(self):
        with self._lock:
            self.monitoring_started_at = time.time()

    def get_adaptive_interval(self) -> int:
        with self._lock:
            if self.monitoring_started_at is None:
                return POLL_INTERVAL_FAST_SECONDS
            elapsed = time.time() - self.monitoring_started_at
            if elapsed < FAST_PHASE_DURATION_SECONDS:
                return POLL_INTERVAL_FAST_SECONDS
            return POLL_INTERVAL_SLOW_SECONDS

    def elapsed_minutes(self) -> int:
        with self._lock:
            if self.monitoring_started_at is None:
                return 0
            return int((time.time() - self.monitoring_started_at) / 60)

    # ── pause ─────────────────────────────────────────────────────────────────
    def pause_until(self, dt: datetime):
        with self._lock:
            self.paused_until = dt

    def resume(self):
        with self._lock:
            self.paused_until = None

    def is_paused(self) -> bool:
        with self._lock:
            if self.paused_until and datetime.now() < self.paused_until:
                return True
            self.paused_until = None
            return False


# ── Telegram Bot ───────────────────────────────────────────────────────────────

class TelegramBot:
    def __init__(self, token: str, primary_chat_id: int):
        self.token = token
        self.primary_chat_id = primary_chat_id
        self.api = f"https://api.telegram.org/bot{token}"
        self.enabled = bool(token and primary_chat_id)
        log.info("✅ Telegram enabled." if self.enabled else "⚠️  Telegram disabled.")

    def send_message(self, chat_id: int, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            r = requests.post(
                f"{self.api}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            return r.status_code == 200
        except Exception as e:
            log.error("Telegram sendMessage: %s", e)
            return False

    def send_photo(self, chat_id: int, image_path: str, caption: str = "") -> bool:
        if not self.enabled:
            return False
        try:
            with open(image_path, "rb") as f:
                r = requests.post(
                    f"{self.api}/sendPhoto",
                    files={"photo": f},
                    data={"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"},
                    timeout=30,
                )
            return r.status_code == 200
        except Exception as e:
            log.error("Telegram sendPhoto: %s", e)
            return False

    def get_updates(self, offset: int | None = None, timeout: int = 30) -> list:
        if not self.enabled:
            return []
        try:
            r = requests.get(
                f"{self.api}/getUpdates",
                params={"timeout": timeout, "offset": offset},
                timeout=timeout + 5,
            )
            if r.status_code == 200:
                return r.json().get("result", [])
        except requests.exceptions.RequestException as e:
            log.error("Telegram getUpdates: %s", e)
        return []


# ── Helper functions ───────────────────────────────────────────────────────────

def trigger_ha_webhook(state: str):
    """Notify Home Assistant of script run state via webhook."""
    if not HA_WEBHOOK_URL:
        return
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        r = requests.post(HA_WEBHOOK_URL, json={"state": state}, timeout=5, verify=False)
        log.info("🌐 Webhook → state='%s' (HTTP %s)", state, r.status_code)
    except Exception as e:
        log.error("Webhook failed: %s", e)


def download_image(filename: str, retries: int = 2) -> bool:
    """Download camera snapshot from Home Assistant. Retries on failure."""
    headers = {"Authorization": f"Bearer {HA_LONG_LIVED_TOKEN}"}
    for attempt in range(1, retries + 2):
        try:
            r = requests.get(HA_CAMERA_IMAGE_URL, headers=headers, timeout=15)
            r.raise_for_status()
            with open(filename, "wb") as f:
                f.write(r.content)
            return True
        except requests.exceptions.RequestException as e:
            log.warning("Image download attempt %d/%d failed: %s", attempt, retries + 1, e)
            if attempt <= retries:
                time.sleep(3)
    return False


def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_vision_analysis(image_path: str, prompt: str, max_tokens: int = 60) -> str | None:
    """Send image + prompt to local Ollama vision model; return response text."""
    log.info("🧠 Requesting vision analysis (max_tokens=%d)…", max_tokens)
    b64 = encode_image_to_base64(image_path)
    payload = {
        "model": VISION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }
    try:
        r = requests.post(
            VISION_SERVER_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=90,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"].strip()
        log.info("💡 Vision response: %s", content)
        return content
    except Exception as e:
        log.error("Vision server error: %s", e)
        return None


def remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


# ── Monitoring loop ────────────────────────────────────────────────────────────

def monitoring_loop(bot: TelegramBot, active: threading.Event, state: MonitorState):
    log.info("🔬 Monitoring thread started.")
    empty_since: float | None = None
    consecutive_stops: int = 0
    consecutive_warns: int = 0
    last_warn_alert: float = 0.0

    while True:
        active.wait()  # blocks here while monitoring is OFF

        if state.is_paused():
            time.sleep(5)
            continue

        elapsed_min = state.elapsed_minutes()
        interval = state.get_adaptive_interval()
        fast_threshold = FAST_PHASE_DURATION_SECONDS // 60

        if elapsed_min < fast_threshold:
            log.info(
                "--- Check @ %s | ⚡ Fast phase (%d/%d min) ---",
                time.strftime("%H:%M:%S"), elapsed_min, fast_threshold,
            )
        else:
            log.info(
                "--- Check @ %s | 🐢 Slow phase (%d min elapsed) ---",
                time.strftime("%H:%M:%S"), elapsed_min,
            )

        if not download_image(LOCAL_IMAGE_FILENAME):
            log.warning("Image download failed; retrying in %ds.", interval)
            time.sleep(interval)
            continue

        result = get_vision_analysis(LOCAL_IMAGE_FILENAME, MONITOR_PROMPT, max_tokens=60)

        if not result:
            log.warning("Analysis failed — no response from vision server.")
            state.set_status("Error: no response from vision server.")
            time.sleep(interval)
            continue

        state.set_status(result)
        upper = result.upper()

        # ── EMPTY ─────────────────────────────────────────────────────────────
        if "EMPTY" in upper:
            consecutive_stops = 0
            consecutive_warns = 0
            if empty_since is None:
                empty_since = time.time()
                log.info("🪣 Bed appears empty — timeout timer started.")
            elif (time.time() - empty_since) > EMPTY_TIMEOUT_SECONDS:
                bot.send_message(
                    bot.primary_chat_id,
                    "✅ Printer bed has been empty for over an hour.\n"
                    "Monitoring stopped. Send /start to resume.",
                )
                active.clear()
                trigger_ha_webhook("off")
                empty_since = None
                remove_file(LOCAL_IMAGE_FILENAME)
            time.sleep(interval)
            continue

        empty_since = None

        # ── STOP ──────────────────────────────────────────────────────────────
        if upper.startswith("STOP"):
            consecutive_warns = 0
            consecutive_stops += 1
            log.warning("🚨 STOP signal %d/%d", consecutive_stops, STOP_CONFIRM_COUNT)
            if consecutive_stops >= STOP_CONFIRM_COUNT:
                log.error("🚨 CRITICAL FAILURE CONFIRMED!")
                bot.send_photo(
                    bot.primary_chat_id,
                    LOCAL_IMAGE_FILENAME,
                    caption=f"‼️ <b>CRITICAL FAILURE</b> ‼️\n\n<code>{result}</code>",
                )
                consecutive_stops = 0
            else:
                log.warning("  Waiting for %d more confirmation(s).", STOP_CONFIRM_COUNT - consecutive_stops)

        # ── WARN ──────────────────────────────────────────────────────────────
        elif upper.startswith("WARN"):
            consecutive_stops = 0
            consecutive_warns += 1
            now = time.time()
            cooldown_over = (now - last_warn_alert) >= WARN_COOLDOWN_SECONDS

            if consecutive_warns >= WARN_ESCALATE_COUNT:
                log.warning("⚠️ Persistent WARN — escalating!")
                bot.send_photo(
                    bot.primary_chat_id,
                    LOCAL_IMAGE_FILENAME,
                    caption=(
                        f"🔴 <b>PERSISTENT ISSUE</b> — {consecutive_warns} consecutive WARNs\n\n"
                        f"<code>{result}</code>"
                    ),
                )
                consecutive_warns = 0
                last_warn_alert = now
            elif cooldown_over:
                log.warning("⚠️ WARN — notifying.")
                bot.send_photo(
                    bot.primary_chat_id,
                    LOCAL_IMAGE_FILENAME,
                    caption=f"⚠️ <b>POTENTIAL ISSUE</b>\n\n<code>{result}</code>",
                )
                last_warn_alert = now
            else:
                remaining = int(WARN_COOLDOWN_SECONDS - (now - last_warn_alert))
                log.info("⚠️ WARN suppressed (cooldown %ds remaining).", remaining)
            remove_file(LOCAL_IMAGE_FILENAME)

        # ── OK ────────────────────────────────────────────────────────────────
        elif upper.startswith("OK"):
            consecutive_stops = 0
            consecutive_warns = 0
            log.info("✅ Print OK.")
            remove_file(LOCAL_IMAGE_FILENAME)

        else:
            log.warning("🤔 Unexpected response: '%s'", result)
            remove_file(LOCAL_IMAGE_FILENAME)

        time.sleep(interval)


# ── Telegram command listener ──────────────────────────────────────────────────

HELP_TEXT = """<b>3D Print Monitor Commands</b>

/start          – Begin monitoring
/stop           – Stop monitoring
/status         – Current status + elapsed time
/describe       – Detailed AI analysis + photo
/snapshot       – Camera photo only (no AI)
/pause <min>   – Pause monitoring for N minutes
/resume         – Resume a paused session
/help           – Show this message
"""


def listen_for_telegram_commands(bot: TelegramBot, active: threading.Event, state: MonitorState):
    log.info("🎧 Telegram command listener started.")
    last_id = 0

    while True:
        updates = bot.get_updates(offset=last_id + 1)
        for update in updates:
            last_id = update["update_id"]
            if "message" not in update or "text" not in update["message"]:
                continue

            msg = update["message"]
            chat_id: int = msg["chat"]["id"]
            text: str = msg["text"].strip()
            cmd = text.lower().split()[0]
            args = text.split()[1:]

            log.info("📨 Command '%s' from chat %s", text, chat_id)

            if cmd == "/start":
                active.set()
                state.resume()
                state.mark_started()
                trigger_ha_webhook("on")
                bot.send_message(
                    chat_id,
                    f"✅ <b>Monitoring started.</b>\n"
                    f"⚡ Fast checks every {POLL_INTERVAL_FAST_SECONDS}s for the first "
                    f"{FAST_PHASE_DURATION_SECONDS // 60} minutes.\n"
                    f"🐢 Then slowing to every {POLL_INTERVAL_SLOW_SECONDS}s.",
                )

            elif cmd == "/stop":
                active.clear()
                trigger_ha_webhook("off")
                bot.send_message(chat_id, "🛑 <b>Monitoring stopped.</b>")

            elif cmd == "/pause":
                minutes = int(args[0]) if args and args[0].isdigit() else 30
                until = datetime.now() + timedelta(minutes=minutes)
                state.pause_until(until)
                bot.send_message(
                    chat_id,
                    f"⏸️ Monitoring paused for <b>{minutes} minute(s)</b>.\n"
                    f"Resumes at {until.strftime('%H:%M:%S')} or send /resume.",
                )

            elif cmd == "/resume":
                state.resume()
                bot.send_message(chat_id, "▶️ Monitoring resumed.")

            elif cmd == "/describe":
                bot.send_message(chat_id, "🔍 Analyzing… this may take a moment.")
                if download_image(DESCRIBE_TEMP_FILENAME):
                    analysis = get_vision_analysis(
                        DESCRIBE_TEMP_FILENAME, ANALYZE_PROMPT, max_tokens=450
                    )
                    if analysis:
                        bot.send_photo(chat_id, DESCRIBE_TEMP_FILENAME)
                        bot.send_message(chat_id, analysis)
                    else:
                        bot.send_message(chat_id, "❌ Vision model returned no response.")
                    remove_file(DESCRIBE_TEMP_FILENAME)
                else:
                    bot.send_message(chat_id, "❌ Could not download camera image.")

            elif cmd == "/snapshot":
                tmp = "snapshot_tmp.jpg"
                if download_image(tmp):
                    bot.send_photo(chat_id, tmp, caption="📸 Live snapshot")
                    remove_file(tmp)
                else:
                    bot.send_message(chat_id, "❌ Could not download camera image.")

            elif cmd == "/status":
                if active.is_set():
                    paused_note = " (⏸️ PAUSED)" if state.is_paused() else ""
                    elapsed = state.elapsed_minutes()
                    interval = state.get_adaptive_interval()
                    fast_threshold = FAST_PHASE_DURATION_SECONDS // 60
                    phase = "⚡ Fast" if elapsed < fast_threshold else "🐢 Slow"
                    last = state.get_status()
                    bot.send_message(
                        chat_id,
                        f"ℹ️ <b>Monitoring ON{paused_note}</b>\n"
                        f"Elapsed: {elapsed} min | {phase} phase | Interval: {interval}s\n\n"
                        f"Last result:\n<code>{last}</code>",
                    )
                else:
                    bot.send_message(chat_id, "ℹ️ <b>Monitoring OFF.</b> Send /start to begin.")

            elif cmd == "/help":
                bot.send_message(chat_id, HELP_TEXT)

            else:
                bot.send_message(
                    chat_id,
                    f"Unknown command: <code>{cmd}</code>\n\nSend /help for the command list.",
                )

        time.sleep(1)


# ── Home Assistant network listener ───────────────────────────────────────────

def listen_for_network_commands(bot: TelegramBot, active: threading.Event, state: MonitorState):
    """Accept raw TCP from Home Assistant on NETWORK_LISTENER_PORT.
    Send b'start' or b'stop' to control monitoring."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", NETWORK_LISTENER_PORT))
        srv.listen()
        log.info("📡 HA network listener on port %d.", NETWORK_LISTENER_PORT)
        while True:
            try:
                conn, addr = srv.accept()
                with conn:
                    data = conn.recv(1024).strip()
                    log.info("📡 HA signal '%s' from %s", data, addr)
                    if data == b"start":
                        active.set()
                        state.resume()
                        state.mark_started()
                        trigger_ha_webhook("on")
                        bot.send_message(
                            bot.primary_chat_id,
                            f"✅ <b>Print started — monitoring active.</b>\n"
                            f"⚡ Fast checks every {POLL_INTERVAL_FAST_SECONDS}s for the first "
                            f"{FAST_PHASE_DURATION_SECONDS // 60} min, "
                            f"then every {POLL_INTERVAL_SLOW_SECONDS}s.",
                        )
                    elif data == b"stop":
                        active.clear()
                        trigger_ha_webhook("off")
                        bot.send_message(
                            bot.primary_chat_id,
                            "🛑 <b>Print finished — monitoring stopped.</b>",
                        )
                    else:
                        log.warning("HA listener: unknown signal '%s'", data)
            except Exception as e:
                log.error("HA listener error: %s", e)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("─── 3D Print AI Monitor ───")

    active_event = threading.Event()
    state = MonitorState()
    bot = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    threads = [
        threading.Thread(
            target=monitoring_loop,
            args=(bot, active_event, state),
            daemon=True, name="monitor",
        ),
        threading.Thread(
            target=listen_for_telegram_commands,
            args=(bot, active_event, state),
            daemon=True, name="telegram",
        ),
        threading.Thread(
            target=listen_for_network_commands,
            args=(bot, active_event, state),
            daemon=True, name="ha-net",
        ),
    ]

    for t in threads:
        t.start()

    log.info("✅ All threads running.")
    log.info("   Model  : %s", VISION_MODEL_NAME)
    log.info("   Ollama : %s", VISION_SERVER_URL)
    log.info(
        "   Polling: %ds (fast, first %d min) → %ds (slow)",
        POLL_INTERVAL_FAST_SECONDS,
        FAST_PHASE_DURATION_SECONDS // 60,
        POLL_INTERVAL_SLOW_SECONDS,
    )
    log.info("Send /help to your Telegram bot to get started.")

    trigger_ha_webhook("on")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("👋 Shutting down…")
    finally:
        trigger_ha_webhook("off")
        for f in (LOCAL_IMAGE_FILENAME, DESCRIBE_TEMP_FILENAME, "snapshot_tmp.jpg"):
            remove_file(f)
