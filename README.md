# Negotiation Coach

Practice salary negotiations against AI personas with voice. Get a scored debrief after every session.

Built with FastAPI, Claude claude-opus-4-7, and Gradium voice AI.

![Setup Screen](static/images/favicon.png)

## What it does

- Paste a job description and pick a difficulty (Easy → Impossible)
- Choose a negotiator persona — Emma, Kent, Eva, or Jack — each with a distinct style and voice
- The AI makes a lowball opening offer; you negotiate back in text or voice
- An offer slide carousel breaks down salary, equity, benefits, and total comp
- At the end, get a coaching debrief: what worked, what to practice next, and a score

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Python |
| AI | Anthropic Claude claude-opus-4-7 (adaptive thinking) |
| Voice TTS/STT | Gradium voice AI |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Storage | Browser localStorage (no database) |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export GRADIUM_API_KEY="..."
```

### 3. Run

```bash
python3 app.py
```

Open [http://localhost:8000](http://localhost:8000).

## API Keys

- **Anthropic** — [console.anthropic.com](https://console.anthropic.com) → API Keys
- **Gradium** — [gradium.ai](https://gradium.ai) → Dashboard

## Project Structure

```
app.py          # FastAPI server — all endpoints and Claude/Gradium logic
index.html      # Single-page frontend — setup, chat, assessment
requirements.txt
static/
  images/       # Negotiator photos + favicon
```

## Features

- **4 difficulty levels** — Easy to Impossible, each with distinct AI behavior
- **4 voice personas** — HR Manager, Senior Recruiter, Director of HR, VP of Talent
- **Voice input** — Custom WAV recorder using Web Audio API (works in all browsers)
- **Offer slides** — 5-card carousel generated from the actual opening offer
- **Session history** — Past negotiations stored in localStorage with scores
- **Coaching assessment** — Strengths, areas to practice, final offer recap
- **Quick replies** — One-tap negotiation phrases for common arguments
- **Sample roles** — Pre-loaded job descriptions to start practicing immediately
