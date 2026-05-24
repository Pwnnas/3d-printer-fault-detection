# 3D Printer Fault Detection

Local AI-based fault detection for 3D printers using YOLOv8, Home Assistant camera snapshots, Telegram alerts, and optional second-stage vision model verification.

The project was started because I could not find a reliable print-failure detection system that worked well with Bambu Lab printers. Unlike more open Klipper-based setups, Bambu Lab printers do not expose the same level of control or integration options, so I wanted a system that could monitor prints from the camera feed and still work independently.

The long-term goal is a lightweight, self-hosted monitoring system that can run on Raspberry Pi-class hardware and integrate cleanly with Home Assistant.

---

## Project status

Work in progress.

Current status:

- YOLOv8 model runs continuously against printer camera snapshots
- Telegram bot sends alerts and status updates
- Start and finish notifications are supported
- Verbose mode can report YOLOv8 model behavior and confidence
- Optional second-stage verification with a larger vision model is supported
- Currently testing YOLOv8-only mode to evaluate the custom model without fallback influence
- Home Assistant integration is planned to be cleaned up and finalized

---

## Why this project exists

Most 3D printer failure detection projects are either:

- built around open printer firmware such as Klipper
- dependent on cloud services
- too heavy for low-power hardware
- trained on generic public images that do not match real camera placement
- prone to false positives when using large general-purpose vision models

I originally tested using only a large vision model running on GPU. It worked, but it was unnecessary for many cases and sometimes less reliable. The larger model could become too detail-oriented and detect failures that were not actually failures.

The current approach is different:

- use a smaller YOLOv8 model for fast local classification
- train on real-world printer camera footage
- optionally ask a larger model only when confidence is low or the result is unclear

This keeps the system faster, lighter, and better suited for Raspberry Pi-class hardware.

---

## Dataset

The custom YOLOv8 model is trained from real print timelapses collected from approximately 300-400 hours of actual printing.

The dataset includes frames from:

- normal successful prints
- failed prints
- suspicious or borderline print states
- real camera angles from modern enclosed 3D printers

Since many modern 3D printers have a fixed camera in one corner of the chamber, the model is trained on images that better match the real deployment environment instead of relying only on public datasets.

The dataset is not currently included in this repository.

---

## Features

- Local YOLOv8-based print monitoring
- Designed for Raspberry Pi or small local servers
- Home Assistant camera snapshot support
- Telegram notifications
- Optional large vision model verification
- Configurable polling intervals
- Warning and stop-condition logic
- Verbose model behavior logging
- Self-hosted and privacy-friendly
- No cloud AI dependency required for normal YOLOv8 operation

---

## Detection flow

```text
Home Assistant camera
        |
        v
Fetch snapshot
        |
        v
YOLOv8 classification
        |
        +--> OK / continue monitoring
        |
        +--> Warning or uncertain state
                  |
                  +--> Optional larger vision model check
                  |
                  v
        Telegram alert / Home Assistant webhook / status update
```

---

## Current operating modes

The system can be configured to run in different modes.

### YOLOv8 only

Runs only the custom YOLOv8 model.

This is the current mode used for testing the newly trained model without influence from a larger fallback model.

### YOLOv8 + verification

Runs YOLOv8 first and uses a larger vision model only when the result is uncertain or potentially serious.

This reduces GPU usage and avoids asking a large model to inspect every frame.

### Vision model only

Useful for testing and comparison, but not the preferred deployment mode.

---

## Notifications

Telegram notifications are used for:

- monitor startup
- monitor shutdown or finished print
- warnings
- possible failures
- verbose model and debug information when enabled

Example alert types:

```text
OK
WARN: possible print issue detected
STOP: likely failed print
```

---

## Home Assistant integration

The project currently uses Home Assistant camera snapshots as input.

Planned Home Assistant improvements:

- automation examples
- webhook support for warning and stop events
- sensor/entity state reporting
- dashboard card examples
- cleaner setup instructions
- optional printer pause/stop automation where supported

The goal is to make the system useful even for printers where direct firmware-level control is limited.

---

## Hardware target

The project is intended to run on:

- Raspberry Pi
- small Linux server
- mini PC
- Proxmox LXC or VM host
- local AI server

The main target is low-power local inference, not cloud-hosted processing.

---

## Tech stack

- Python
- YOLOv8 / Ultralytics
- Home Assistant
- Telegram Bot API
- dotenv-based configuration
- Optional OpenAI-compatible vision endpoint
- Optional Ollama/local vision model backend

---

## Configuration

Configuration is handled through environment variables.

Example values:

```env
# Home Assistant
HA_CAMERA_IMAGE_URL=
HA_LONG_LIVED_TOKEN=
HA_WEBHOOK_URL=

# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# YOLOv8
YOLO_CLS_MODEL_PATH=
YOLO_CLS_CONFIDENCE=0.65
YOLO_CLS_DEFECT_THRESHOLD=0.75

# Polling
POLL_INTERVAL_FAST_SECONDS=20
POLL_INTERVAL_SLOW_SECONDS=120
FAST_PHASE_DURATION_SECONDS=1800

# Warning and stop logic
STOP_CONFIRM_COUNT=2
WARN_ESCALATE_COUNT=5
WARN_COOLDOWN_SECONDS=600

# Optional vision model backend
VISION_SERVER_URL=http://127.0.0.1:11434/v1/chat/completions
```

A full `.env.example` will be added as the configuration stabilizes.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Pwnnas/3d-printer-fault-detection.git
cd 3d-printer-fault-detection
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your Home Assistant, Telegram, and model settings.

Run the monitor:

```bash
python main.py
```

Note: command names and setup steps may change while the project is still under active development.

---

## Raspberry Pi goal

A major design goal is to make the system practical on Raspberry Pi-class hardware.

The reason for this is simple: a 3D printer monitoring system should not require a large GPU server just to detect common print failures. The YOLOv8 model is intended to handle the normal monitoring loop locally, while larger models can be used only when needed.

Target behavior:

- low CPU usage during idle or normal print states
- fast enough detection for practical monitoring
- no cloud dependency for normal operation
- simple deployment on a small Linux device

---

## Design goals

The project is built around a few practical constraints:

- run locally
- avoid unnecessary GPU usage
- work with printers that do not expose open firmware control
- reduce false positives
- use real printer camera perspectives
- be simple enough to deploy on a Raspberry Pi
- integrate with Home Assistant instead of replacing it
- keep alerts understandable and actionable

---

## Roadmap

- [ ] Finalize YOLOv8 model testing
- [ ] Improve README and setup documentation
- [ ] Add `.env.example`
- [ ] Add Raspberry Pi deployment guide
- [ ] Add Home Assistant automation examples
- [ ] Add systemd service example
- [ ] Add Docker or container deployment option
- [ ] Add example Telegram alert screenshots
- [ ] Add example camera images with detections
- [ ] Improve model confidence logging
- [ ] Add clearer WARN/STOP state handling
- [ ] Add dashboard examples for Home Assistant
- [ ] Document model training process
- [ ] Document dataset generation workflow from timelapses

---

## Limitations

This project is not intended to be the only safety system for a 3D printer.

It should be treated as an additional monitoring layer, not a replacement for:

- proper printer maintenance
- smoke detection
- thermal runaway protection
- safe printer placement
- human supervision for risky prints

Computer vision can produce false positives and false negatives.

---

## Safety note

This project can help detect visual print failures, but it should not be relied on as a fire-safety system.

For unattended printing, use proper hardware safety measures such as smoke detectors, safe printer placement, and manufacturer-supported safety features.

---

## Motivation

The main motivation is to build a practical, local, privacy-friendly failure detection system that works with the cameras already built into modern 3D printers.

Instead of relying on generic image datasets or cloud-based analysis, this project focuses on real print footage, fixed chamber camera angles, and lightweight inference that can run on affordable hardware.

The project also explores a hybrid approach where a small model does the constant monitoring and a larger model is only used when it adds value.

---

## Author

Built by [Pwnnas](https://github.com/Pwnnas) as a practical homelab, AI, and 3D-printing automation project.
