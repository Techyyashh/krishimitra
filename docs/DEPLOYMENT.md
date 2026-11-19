# 🚀 Deployment guide

Goal: a public HTTPS URL that serves the web app **and** the API, with the model on the Hugging Face Hub,
and automatic redeploys whenever you push to `main`. Everything here is free.

```
GitHub (code) ──CI passes──► GitHub Actions ──upload──► Hugging Face Space (Docker)
                                                          │  downloads model at startup
Hugging Face Hub (model: krishimitra.onnx) ◄──────────────┘
```

## 0. Before you start
- [ ] Model trained and exported: `models/krishimitra.onnx` + `models/classes.json`
- [ ] `pytest` passes locally
- [ ] App works locally: `uvicorn api.main:app --reload`
- [ ] Free accounts: [huggingface.co](https://huggingface.co), GitHub

## 1. Publish the model on the Hugging Face Hub
1. Hugging Face → Settings → Access Tokens → **New token** (type: *Write*). Keep it secret.
2. Fill in `models/MODEL_CARD.md` with your real metrics.
3. Upload:
   ```powershell
   $env:HF_TOKEN="hf_xxx"
   python scripts/upload_model.py --repo Techyyashh/krishimitra-model
   ```
4. Check `https://huggingface.co/Techyyashh/krishimitra-model`.

## 2. Create the Space (hosting)
First deploy from your laptop:
```powershell
$env:HF_TOKEN="hf_xxx"; $env:HF_SPACE_ID="Techyyashh/krishimitra"
python scripts/deploy_hf_space.py --dry-run   # see what will be uploaded
python scripts/deploy_hf_space.py
```
Then on the Space page → **Settings**:
| Type | Name | Value |
|---|---|---|
| Variable | `HF_MODEL_REPO` | `Techyyashh/krishimitra-model` |
| Secret | `ANTHROPIC_API_KEY` | your Claude API key (optional) |
| Variable | `ALLOWED_ORIGINS` | your site URL, e.g. `https://techyyashh-krishimitra.hf.space` |

The Space builds the Docker image (≈3–5 min). Your app is live at
`https://techyyashh-krishimitra.hf.space` and API docs at `/api/docs`.

> **Free-tier sleep:** free Spaces sleep after ~48 h without visitors and take ~1 min to wake up.
> UptimeRobot pings (step 5) keep it awake during your demo period.

> **Storage:** the SQLite stats database on a free Space is lost when the Space restarts.
> That's fine for a student project. For permanent data, move feedback to a free PostgreSQL
> database (Supabase or Neon) later.

## 3. Automatic deploys (CI/CD)
GitHub repo → Settings → Secrets and variables → Actions:
- **Secret** `HF_TOKEN` = your write token
- **Variable** `HF_SPACE_ID` = `Techyyashh/krishimitra`

Now every push to `main`: `CI` runs lint + tests + Docker build → if green, `Deploy` uploads the app to the Space.

## 4. Custom domain (optional, ~₹500–900/year)
Hugging Face Spaces can't use a custom domain on the free plan. Options:
- Keep the free `*.hf.space` URL (fine for a project, the Play Store and your resume), **or**
- Deploy the same Docker image on **Render** (free web service, custom domains supported):
  New → Web Service → connect GitHub repo → Runtime: Docker → add the same environment variables →
  Settings → Custom Domain. Point the domain's DNS to Render (Cloudflare free DNS works well).

## 5. Monitoring (free)
| Tool | Setup |
|---|---|
| **UptimeRobot** | New monitor → HTTP(s) → `https://…hf.space/api/health` every 5 min → e-mail alert |
| **Usage** | `GET /api/stats` – total scans, top diseases, feedback accuracy, most-reported-wrong classes |
| **Logs** | Space → *Logs* tab (startup errors, LLM fallbacks, rate limits) |
| **LLM cost** | console.anthropic.com → set a monthly spend limit and e-mail alerts |

Optional: add [Sentry](https://sentry.io) (free tier) for error tracking – `pip install sentry-sdk[fastapi]`
and call `sentry_sdk.init(dsn=os.environ["SENTRY_DSN"])` at the top of `api/main.py`.

## 6. Production checklist
- [ ] `ALLOWED_ORIGINS` set to your real domain(s)
- [ ] `ANTHROPIC_API_KEY` stored only as a Space secret (never in code or the frontend)
- [ ] Spend limit set on the Claude API account
- [ ] `/api/health` returns `"model_loaded": true`
- [ ] Privacy page (`/privacy.html`) filled in with your contact e-mail and date
- [ ] Tested on a real low-end Android phone on mobile data
- [ ] Lighthouse (Chrome DevTools) → PWA installable, Performance & Accessibility ≥ 90

## Monthly maintenance
1. Look at `/api/stats` → `most_reported_wrong`.
2. Collect more photos of those classes (field visits, KVK), add them to training data.
3. Retrain on Colab → export → `upload_model.py` → restart the Space (it re-downloads the model).
4. Bump `project.version` in `configs/config.yaml` and create a GitHub release.
