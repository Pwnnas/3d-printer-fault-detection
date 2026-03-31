"""
3D Print AI Monitor — YOLOv8n Edition
Monitors a 3D printer via Home Assistant camera.
  • STOP / WARN / OK decisions → fast local YOLOv8n (best.pt)
  • /describe command          → detailed Ollama/LLaVA analysis (unchanged)
  • Alerts                     → Telegram
SETUP:
  pip install requests python-dotenv ultralytics opencv-python
  pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117
  pip install "numpy<2.0"
  Copy .env.example to .env and add YOLO_MODEL_PATH.
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

# Home Assistant
HA_CAMERA_IMAGE_URL     = os.getenv("HA_CAMERA_IMAGE_URL")
HA_LONG_LIVED_TOKEN     = os.getenv("HA_LONG_LIVED_TOKEN")
HA_WEBHOOK_URL          = os.getenv("HA_WEBHOOK_URL")

# YOLOv8n — your trained model
YOLO_MODEL_PATH         = os.getenv(
    "YOLO_MODEL_PATH",
    r"C:\train\runs\detect\obico_yolov8n\weights\best.pt",
)
# Confidence thresholds — tune these after watching false-positive rates
YOLO_STOP_CONF          = float(os.getenv("YOLO_STOP_CONF", 0.65))   # very confident = STOP
YOLO_WARN_CONF          = float(os.getenv("YOLO_WARN_CONF", 0.35))   # moderate = WARN
# Classes that should trigger STOP vs WARN (by name, lowercase)
STOP_CLASSES            = {"spaghetti", "error"}
WARN_CLASSES            = {"extrusor", "part"}

# Vision server (Ollama) — only used by /describe
VISION_SERVER_IP        = os.getenv("VISION_SERVER_IP", "127.0.0.1")
VISION_SERVER_PORT      = int(os.getenv("VISION_SERVER_PORT", 11434))
VISION_SERVER_URL       = f"http://{VISION_SERVER_IP}:{VISION_SERVER_PORT}/v1/chat/completions"
VISION_MODEL_NAME       = os.getenv("VISION_MODEL_NAME", "llava:7b-v1.6")

# Telegram
TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID        = int(os.getenv("TELEGRAM_CHAT_ID", 0))

# Monitoring behaviour
POLL_INTERVAL_FAST_SECONDS   = int(os.getenv("POLL_INTERVAL_FAST_SECONDS", 20))
POLL_INTERVAL_SLOW_SECONDS   = int(os.getenv("POLL_INTERVAL_SLOW_SECONDS", 120))
FAST_PHASE_DURATION_SECONDS  = int(os.getenv("FAST_PHASE_DURATION_SECONDS", 1800))
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

# ── Load YOLOv8n model (once at startup) ──────────────────────────────────────
try:
    from ultralytics import YOLO
    import torch
    import cv2
except ImportError as e:
    print(f"\n[ERROR] Missing package: {e}")
    print("        pip install ultralytics opencv-python\n")
    raise SystemExit(1)

if not os.path.isfile(YOLO_MODEL_PATH):
    print(f"\n[ERROR] YOLO model not found: {YOLO_MODEL_PATH}")
    print("        Set YOLO_MODEL_PATH in your .env or check C:\\train\\runs\\...\n")
    raise SystemExit(1)

log.info("Loading YOLOv8n model: %s", YOLO_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_device = 0 if torch.cuda.is_available() else "cpu"
log.info("YOLOv8n running on: %s", "GPU" if yolo_device == 0 else "CPU")

# ── YOLOv8n inference ──────────────────────────────────────────────────────────

def run_yolo(image_path: str) -> tuple[str, str, list]:
    """
    Run YOLOv8n on image_path.

    Returns:
        decision  : "STOP", "WARN", "OK", or "OK EMPTY"
        reason    : human-readable string summarising detections
        detections: list of dicts [{class, confidence, bbox}, …]
    """
    results     = yolo_model.predict(
        source=image_path,
        conf=YOLO_WARN_CONF,   # collect everything above the lower WARN threshold
        device=yolo_device,
        verbose=False,
        save=False,
    )
    boxes       = results[0].boxes
    class_names = yolo_model.names

    if len(boxes) == 0:
        return "OK", "No failures detected.", []

    detections  = []
    stop_hits   = []
    warn_hits   = []

    for box in boxes:
        cls_name   = class_names[int(box.cls[0])].lower()
        confidence = float(box.conf[0])
        bbox       = [int(v) for v in box.xyxy[0]]

        detections.append({"class": cls_name, "confidence": confidence, "bbox": bbox})

        if cls_name in STOP_CLASSES and confidence >= YOLO_STOP_CONF:
            stop_hits.append(f"{cls_name} ({confidence:.0%})")
        elif cls_name in WARN_CLASSES or confidence >= YOLO_WARN_CONF:
            warn_hits.append(f"{cls_name} ({confidence:.0%})")

    if stop_hits:
        reason = "Detected: " + ", ".join(stop_hits)
        return "STOP", reason, detections
    if warn_hits:
        reason = "Detected: " + ", ".join(warn_hits)
        return "WARN", reason, detections

    return "OK", "Low-confidence detections only.", detections


def save_annotated(image_path: str, out_path: str):
    """Draw YOLO bounding boxes and save to out_path."""
    try:
        results         = yolo_model.predict(
            source=image_path, conf=YOLO_WARN_CONF,
            device=yolo_device, verbose=False, save=False,
        )
        annotated_frame = results[0].plot()
        cv2.imwrite(out_path, annotated_frame)
    except Exception as e:
        log.warning("Could not save annotated image: %s", e)
        # Fall back to unannotated original
        import shutil
        shutil.copy(image_path, out_path)

# ── Describe prompt (LLaVA — unchanged) ────────────────────────────────────────
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

# ── Shared State ───────────────────────────────────────────────────────────────
class MonitorState:
    def __init__(self):
        self._lock = threading.Lock()
        self.latest_status: str = "Unknown (monitoring not yet run)"
        self.paused_until: datetime | None = None
        self.monitoring_started_at: float | None = None

    def get_status(self) -> str:
        with self._lock:
            return self.latest_status

    def set_status(self, s: str):
        with self._lock:
            self.latest_status = s

    def mark_started(self):
        with self._lock:
            self.monitoring_started_at = time.time()

    def get_adaptive_interval(self) -> int:
        with self._lock:
            if self.monitoring_started_at is None:
                return POLL_INTERVAL_FAST_SECONDS
            elapsed = time.time() - self.monitoring_started_at
            return POLL_INTERVAL_FAST_SECONDS if elapsed < FAST_PHASE_DURATION_SECONDS else POLL_INTERVAL_SLOW_SECONDS

    def elapsed_minutes(self) -> int:
        with self._lock:
            if self.monitoring_started_at is None:
                return 0
            return int((time.time() - self.monitoring_started_at) / 60)

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
        self.token          = token
        self.primary_chat_id = primary_chat_id
        self.api            = f"https://api.telegram.org/bot{token}"
        self.enabled        = bool(token and primary_chat_id)
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
    if not HA_WEBHOOK_URL:
        return
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    try:
        r = requests.post(HA_WEBHOOK_URL, json={"state": state}, timeout=5, verify=False)
        log.info("🌐 Webhook → state='%s' (HTTP %s)", state, r.status_code)
    except Exception as e:
        log.error("Webhook failed: %s", e)

def download_image(filename: str, retries: int = 2) -> bool:
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

def get_vision_analysis(image_path: str, prompt: str, max_tokens: int = 60) -> str | None:
    """LLaVA analysis — only called by /describe."""
    log.info("🧠 Requesting LLaVA analysis…")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    payload = {
        "model": VISION_MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
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
        log.info("💡 LLaVA: %s", content[:120])
        return content
    except Exception as e:
        log.error("Vision server error: %s", e)
        return None

def remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass

# ── Monitoring loop (now powered by YOLOv8n) ──────────────────────────────────
def monitoring_loop(bot: TelegramBot, active: threading.Event, state: MonitorState):
    log.info("🔬 Monitoring thread started (YOLOv8n mode).")
    empty_since: float | None = None
    consecutive_stops: int = 0
    consecutive_warns: int = 0
    last_warn_alert: float = 0.0

    while True:
        active.wait()
        if state.is_paused():
            time.sleep(5)
            continue

        elapsed_min = state.elapsed_minutes()
        interval    = state.get_adaptive_interval()
        fast_thresh = FAST_PHASE_DURATION_SECONDS // 60
        phase       = f"⚡ Fast ({elapsed_min}/{fast_thresh} min)" if elapsed_min < fast_thresh else f"🐢 Slow ({elapsed_min} min)"
        log.info("--- Check @ %s | %s ---", time.strftime("%H:%M:%S"), phase)

        if not download_image(LOCAL_IMAGE_FILENAME):
            log.warning("Image download failed; retrying in %ds.", interval)
            time.sleep(interval)
            continue

        # ── YOLOv8n decision (replaces LLaVA monitor prompt) ─────────────────
        decision, reason, detections = run_yolo(LOCAL_IMAGE_FILENAME)
        status_str = f"{decision} {reason}".strip()
        state.set_status(status_str)
        log.info("🎯 YOLO → %s", status_str)

        # ── EMPTY ─────────────────────────────────────────────────────────────
        if decision == "OK EMPTY":
            consecutive_stops = 0
            consecutive_warns = 0
            if empty_since is None:
                empty_since = time.time()
                log.info("🪣 Bed appears empty — timeout timer started.")
            elif (time.time() - empty_since) > EMPTY_TIMEOUT_SECONDS:
                bot.send_message(
                    bot.primary_chat_id,
                    "✅ Printer bed empty for over an hour.\nMonitoring stopped. Send /start to resume.",
                )
                active.clear()
                trigger_ha_webhook("off")
                empty_since = None
                remove_file(LOCAL_IMAGE_FILENAME)
            time.sleep(interval)
            continue

        empty_since = None

        # ── STOP ──────────────────────────────────────────────────────────────
        if decision == "STOP":
            consecutive_warns = 0
            consecutive_stops += 1
            log.warning("🚨 STOP signal %d/%d — %s", consecutive_stops, STOP_CONFIRM_COUNT, reason)

            if consecutive_stops >= STOP_CONFIRM_COUNT:
                log.error("🚨 CRITICAL FAILURE CONFIRMED — sending annotated alert.")
                annotated = "alert_annotated.jpg"
                save_annotated(LOCAL_IMAGE_FILENAME, annotated)
                caption = (
                    f"‼️ <b>CRITICAL FAILURE CONFIRMED</b> ‼️\n\n"
                    f"<code>{reason}</code>\n\n"
                    f"Detections: {len(detections)} object(s)"
                )
                bot.send_photo(bot.primary_chat_id, annotated, caption=caption)
                remove_file(annotated)
                consecutive_stops = 0
            else:
                log.warning("  Waiting for %d more confirmation(s).", STOP_CONFIRM_COUNT - consecutive_stops)

        # ── WARN ──────────────────────────────────────────────────────────────
        elif decision == "WARN":
            consecutive_stops = 0
            consecutive_warns += 1
            now          = time.time()
            cooldown_ok  = (now - last_warn_alert) >= WARN_COOLDOWN_SECONDS

            if consecutive_warns >= WARN_ESCALATE_COUNT:
                log.warning("⚠️ Persistent WARN × %d — escalating!", consecutive_warns)
                annotated = "warn_annotated.jpg"
                save_annotated(LOCAL_IMAGE_FILENAME, annotated)
                bot.send_photo(
                    bot.primary_chat_id, annotated,
                    caption=(
                        f"🔴 <b>PERSISTENT ISSUE</b> — {consecutive_warns} consecutive WARNs\n\n"
                        f"<code>{reason}</code>"
                    ),
                )
                remove_file(annotated)
                consecutive_warns = 0
                last_warn_alert = now
            elif cooldown_ok:
                log.warning("⚠️ WARN — notifying.")
                annotated = "warn_annotated.jpg"
                save_annotated(LOCAL_IMAGE_FILENAME, annotated)
                bot.send_photo(
                    bot.primary_chat_id, annotated,
                    caption=f"⚠️ <b>POTENTIAL ISSUE DETECTED</b>\n\n<code>{reason}</code>",
                )
                remove_file(annotated)
                last_warn_alert = now
            else:
                remaining = int(WARN_COOLDOWN_SECONDS - (now - last_warn_alert))
                log.info("⚠️ WARN suppressed (cooldown %ds remaining).", remaining)

        # ── OK ────────────────────────────────────────────────────────────────
        elif decision == "OK":
            consecutive_stops = 0
            consecutive_warns = 0
            log.info("✅ Print OK.")

        remove_file(LOCAL_IMAGE_FILENAME)
        time.sleep(interval)

# ── Telegram command listener ──────────────────────────────────────────────────
HELP_TEXT = """<b>3D Print Monitor Commands</b>
/start          – Begin monitoring
/stop           – Stop monitoring
/status         – Current status + elapsed time
/describe       – Detailed LLaVA AI analysis + photo
/snapshot       – Camera photo with YOLO boxes (no text analysis)
/pause &lt;min&gt;    – Pause monitoring for N minutes
/resume         – Resume a paused session
/help           – Show this message
"""

def listen_for_telegram_commands(bot: TelegramBot, active: threading.Event, state: MonitorState):
    log.info("🎧 Telegram command listener started.")
    last_id = 0
    while True:
        updates = bot.get_updates(offset=last_id + 1)
        for update in updates:
            last_id  = update["update_id"]
            if "message" not in update or "text" not in update["message"]:
                continue
            msg      = update["message"]
            chat_id  = msg["chat"]["id"]
            text     = msg["text"].strip()
            cmd      = text.lower().split()[0]
            args     = text.split()[1:]

            log.info("📨 Command '%s' from chat %s", text, chat_id)

            if cmd == "/start":
                active.set()
                state.resume()
                state.mark_started()
                trigger_ha_webhook("on")
                bot.send_message(
                    chat_id,
                    f"✅ <b>Monitoring started (YOLOv8n mode).</b>\n"
                    f"⚡ Fast checks every {POLL_INTERVAL_FAST_SECONDS}s for {FAST_PHASE_DURATION_SECONDS // 60} min.\n"
                    f"🐢 Then every {POLL_INTERVAL_SLOW_SECONDS}s.",
                )
            elif cmd == "/stop":
                active.clear()
                trigger_ha_webhook("off")
                bot.send_message(chat_id, "🛑 <b>Monitoring stopped.</b>")
            elif cmd == "/pause":
                minutes = int(args[0]) if args and args[0].isdigit() else 30
                until   = datetime.now() + timedelta(minutes=minutes)
                state.pause_until(until)
                bot.send_message(
                    chat_id,
                    f"⏸️ Paused for <b>{minutes} minute(s)</b>. Resumes at {until.strftime('%H:%M:%S')}.",
                )
            elif cmd == "/resume":
                state.resume()
                bot.send_message(chat_id, "▶️ Monitoring resumed.")

            elif cmd == "/snapshot":
                # Live snapshot with YOLO bounding boxes drawn on it
                tmp = "snapshot_tmp.jpg"
                if download_image(tmp):
                    annotated = "snapshot_annotated.jpg"
                    save_annotated(tmp, annotated)
                    decision, reason, detections = run_yolo(tmp)
                    caption = f"📸 Live snapshot\n🎯 YOLO: <code>{decision} — {reason}</code>"
                    if detections:
                        caption += f"\n{len(detections)} detection(s)"
                    bot.send_photo(chat_id, annotated, caption=caption)
                    remove_file(tmp)
                    remove_file(annotated)
                else:
                    bot.send_message(chat_id, "❌ Could not download camera image.")

            elif cmd == "/describe":
                # Full LLaVA descriptive analysis (unchanged from original)
                bot.send_message(chat_id, "🔍 Analyzing with LLaVA… this may take a moment.")
                if download_image(DESCRIBE_TEMP_FILENAME):
                    analysis = get_vision_analysis(DESCRIBE_TEMP_FILENAME, ANALYZE_PROMPT, max_tokens=450)
                    decision, reason, _ = run_yolo(DESCRIBE_TEMP_FILENAME)
                    annotated = "describe_annotated.jpg"
                    save_annotated(DESCRIBE_TEMP_FILENAME, annotated)
                    bot.send_photo(chat_id, annotated)
                    remove_file(annotated)
                    if analysis:
                        bot.send_message(
                            chat_id,
                            f"🎯 <b>YOLO:</b> <code>{decision} — {reason}</code>\n\n{analysis}"
                        )
                    else:
                        bot.send_message(chat_id, f"🎯 <b>YOLO:</b> <code>{decision} — {reason}</code>\n\n(LLaVA unavailable)")
                    remove_file(DESCRIBE_TEMP_FILENAME)
                else:
                    bot.send_message(chat_id, "❌ Could not download camera image.")

            elif cmd == "/status":
                if active.is_set():
                    paused_note = " (⏸️ PAUSED)" if state.is_paused() else ""
                    elapsed     = state.elapsed_minutes()
                    interval    = state.get_adaptive_interval()
                    fast_thresh = FAST_PHASE_DURATION_SECONDS // 60
                    phase       = "⚡ Fast" if elapsed < fast_thresh else "🐢 Slow"
                    bot.send_message(
                        chat_id,
                        f"ℹ️ <b>Monitoring ON{paused_note}</b> | {phase} | {interval}s interval\n"
                        f"Elapsed: {elapsed} min\n\nLast YOLO result:\n<code>{state.get_status()}</code>",
                    )
                else:
                    bot.send_message(chat_id, "ℹ️ <b>Monitoring OFF.</b> Send /start to begin.")
            elif cmd == "/help":
                bot.send_message(chat_id, HELP_TEXT)
            else:
                bot.send_message(chat_id, f"Unknown command: <code>{cmd}</code>\n\nSend /help for the list.")
        time.sleep(1)

# ── Home Assistant network listener (unchanged) ────────────────────────────────
def listen_for_network_commands(bot: TelegramBot, active: threading.Event, state: MonitorState):
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
                            f"✅ <b>Print started — YOLOv8n monitoring active.</b>",
                        )
                    elif data == b"stop":
                        active.clear()
                        trigger_ha_webhook("off")
                        bot.send_message(bot.primary_chat_id, "🛑 <b>Print finished — monitoring stopped.</b>")
                    else:
                        log.warning("HA listener: unknown signal '%s'", data)
            except Exception as e:
                log.error("HA listener error: %s", e)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("─── 3D Print AI Monitor — YOLOv8n Edition ───")

    active_event = threading.Event()
    state        = MonitorState()
    bot          = TelegramBot(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    threads = [
        threading.Thread(target=monitoring_loop,              args=(bot, active_event, state), daemon=True, name="monitor"),
        threading.Thread(target=listen_for_telegram_commands, args=(bot, active_event, state), daemon=True, name="telegram"),
        threading.Thread(target=listen_for_network_commands,  args=(bot, active_event, state), daemon=True, name="ha-net"),
    ]
    for t in threads:
        t.start()

    log.info("✅ All threads running.")
    log.info("   YOLO model : %s", YOLO_MODEL_PATH)
    log.info("   STOP conf  : %.0f%% | Classes: %s", YOLO_STOP_CONF * 100, STOP_CLASSES)
    log.info("   WARN conf  : %.0f%% | Classes: %s", YOLO_WARN_CONF * 100, WARN_CLASSES)
    log.info("   Polling    : %ds fast → %ds slow", POLL_INTERVAL_FAST_SECONDS, POLL_INTERVAL_SLOW_SECONDS)
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
