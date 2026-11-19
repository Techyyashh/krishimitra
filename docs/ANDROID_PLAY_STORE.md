# 📱 Android app on Google Play (Trusted Web Activity)

We wrap the live PWA in a **Trusted Web Activity (TWA)** using Google's **Bubblewrap** tool.
The Android app opens your website full-screen, without a browser bar. One codebase for web and Android,
and every website update reaches the app instantly. No Play Store re-review needed.

> ⏱️ **Start early.** New *personal* Play developer accounts must run a **closed test with at least 12 testers
> for 14 continuous days** before they can publish to production (check the current rule in Play Console → Dashboard).
> Identity verification can also take a few days. Create the account in **Week 3–4**.

💰 **Cost:** Google Play developer registration – **US $25 one-time**.

## 1. Prerequisites
- Your PWA is live on HTTPS (see `DEPLOYMENT.md`), e.g. `https://techyyashh-krishimitra.hf.space`
- Chrome DevTools → Lighthouse → "Installable" passes
- Install **Node.js LTS** and **JDK 17**:
  ```powershell
  winget install OpenJS.NodeJS.LTS
  winget install EclipseAdoptium.Temurin.17.JDK
  npm install -g @bubblewrap/cli
  ```

## 2. Create the Android project
```powershell
cd mobile/twa
bubblewrap init --manifest https://techyyashh-krishimitra.hf.space/manifest.webmanifest
```
Answer the questions (see `mobile/twa/twa-settings.md` for recommended values):
- Application ID: `com.techyyashh.krishimitra` (cannot be changed later!)
- App name: `KrishiMitra`, launcher name `KrishiMitra`
- Theme colour `#1f7a3a`, background `#f4f7f2`
- Let Bubblewrap **create a new signing key**. Save `android.keystore` and its passwords somewhere safe.
  **If you lose the key you can never update the app.** Never commit it to Git (it's in `.gitignore`).

## 3. Build
```powershell
bubblewrap build
```
Produces `app-release-bundle.aab` (upload to Play) and `app-release-signed.apk` (install on your phone to test).

## 4. Digital Asset Links (removes the browser bar)
1. Get your key's SHA-256 fingerprint:
   ```powershell
   bubblewrap fingerprint list
   ```
   After the first upload, also copy the **Play App Signing** SHA-256 from Play Console → Setup → App integrity.
2. Copy `mobile/twa/assetlinks.template.json` to `web/.well-known/assetlinks.json`, put in your package name
   and **both** fingerprints, commit and deploy.
3. Check: `https://techyyashh-krishimitra.hf.space/.well-known/assetlinks.json` returns the JSON.

## 5. Play Console
1. [play.google.com/console](https://play.google.com/console) → create developer account (personal) → pay $25 → verify identity.
2. **Create app** → name *KrishiMitra – Crop Doctor*, free, app.
3. **Store listing** (English + add Hindi translation):
   - Short description (80 chars): *Click a leaf photo, get crop disease advice in Hindi & English.*
   - Full description: problem → how it works → features → disclaimer (copy from README)
   - Icon 512×512 (`web/icons/icon-512.png`), feature graphic 1024×500, at least 2 phone screenshots
4. **App content:**
   - Privacy policy URL → `https://…/privacy.html`
   - Ads: No · Target audience: 18+ · Content rating questionnaire
   - **Data safety:** photos processed, not stored; anonymous diagnostics; data encrypted in transit (HTTPS); no data sold
5. **Testing → Closed testing** → create track → upload `.aab` → add ≥ 12 testers by e-mail (friends, classmates)
   → share the opt-in link → testers install and keep it for 14 days.
6. **Production** → apply for access after the closed test → roll out.

## 6. Common review rejections & fixes
| Rejection | Fix |
|---|---|
| Broken functionality / login issues | Make sure the Space isn't asleep during review (UptimeRobot) |
| Missing privacy policy | Privacy URL must load and mention the data you process |
| Misleading claims | Don't claim "100% accurate" or "cures"; keep the disclaimer visible |
| Browser bar visible | `assetlinks.json` missing / wrong fingerprint (use the Play App Signing key) |

## 7. Updating
- Website changes → deploy as usual, the app updates automatically.
- Icon/name/colour changes → bump `appVersionCode` in `twa-manifest.json`, `bubblewrap update && bubblewrap build`, upload the new `.aab`.
