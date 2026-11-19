# Recommended answers for `bubblewrap init`

| Question | Value |
|---|---|
| Domain | `techyyashh-krishimitra.hf.space` (or your custom domain) |
| URL path | `/` |
| Application name | `KrishiMitra – Crop Doctor` |
| Short name | `KrishiMitra` |
| Application ID | `com.techyyashh.krishimitra` (lowercase, permanent) |
| Starting version code | `1` |
| Display mode | `standalone` |
| Orientation | `portrait` |
| Status bar colour | `#1f7a3a` |
| Splash screen colour | `#f4f7f2` |
| Icon URL | `https://…/icons/icon-512.png` |
| Maskable icon URL | `https://…/icons/maskable-512.png` |
| Monochrome icon | skip |
| Shortcuts | skip |
| Include support for Play Billing | No |
| Request geolocation permission | No |
| Key store location | `./android.keystore` (keep a backup outside the repo!) |
| Key name | `krishimitra` |

Files generated here (`app/`, `*.keystore`, `*.aab`, `*.apk`) are git-ignored on purpose.
Commit only `twa-manifest.json` (no secrets inside) so you can rebuild later.
