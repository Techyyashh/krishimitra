// Where the API lives.
// "" = same server that serves this page (default when FastAPI serves the web app).
// If you host the web app separately (e.g. Vercel), set your backend URL, for example:
// window.KRISHIMITRA_API_BASE = "https://your-username-krishimitra.hf.space";
window.KRISHIMITRA_API_BASE = "";

// "server"  = the FastAPI backend runs the AI (full features: AI chat, Claude advice, feedback).
// "browser" = the model runs on the visitor's phone with onnxruntime-web (free static hosting).
//             scripts/build_static.py sets this automatically.
window.KRISHIMITRA_MODE = "server";
