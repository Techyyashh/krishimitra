# 🎤 Launch & career kit

Replace every `–` / `[ ]` with your **real** numbers before you use these texts.

## 2-minute pitch
> "Indian farmers lose up to 40% of their crops to diseases, mostly because they are spotted too late and experts
> are far away. KrishiMitra turns any cheap Android phone into a crop doctor. The farmer clicks a photo of a leaf;
> our EfficientNet model, trained on 54,000 images and tested on real field photos, identifies the disease in under
> [ ] seconds. Instead of guessing, it says 'not sure, retake the photo' when confidence is low. It then explains
> the problem and the treatment in simple Hindi, and reads it aloud, because many farmers prefer listening to reading.
> The advice comes from an LLM that is only allowed to use our verified knowledge base and never invents pesticide
> doses. It always points the farmer to the local KVK. It is live on the web and on the Play Store, it is fully
> open source, and every 'was this correct?' tap helps us improve the model. Next, we want to test it with [ ]
> farmers through our local KVK and add Marathi and offline mode."

## Resume bullets
- Built **KrishiMitra**, an AI crop-disease detector (EfficientNet-B0, PyTorch → ONNX) reaching **[ ]% accuracy on 38 classes** and **[ ]% on real-field PlantDoc images**, with <[ ] ms CPU inference.
- Designed a **grounded LLM advisory pipeline** (Claude API + RAG + static fallback) delivering safe, Hindi/English, voice-enabled treatment advice, and shipped it as a **PWA and Play Store app**.
- Deployed with **FastAPI, Docker, GitHub Actions CI/CD and Hugging Face**, with rate limiting, privacy-by-design analytics and a user-feedback loop for continuous retraining.

## LinkedIn launch post (template)
> 🌾 I just launched **KrishiMitra**, an AI app that helps farmers detect crop diseases from a single leaf photo, with advice in Hindi and voice.
>
> 🔍 38 diseases across 14 crops · 📱 Works on low-end phones · 🗣️ Hindi + English + audio · 🔓 Open source
>
> What I learned: why 99% accuracy in the lab can mean ~[ ]% in a real field, how to stop an LLM from inventing pesticide doses, and how to ship a PWA to the Play Store.
>
> 🌐 Try it: [link] · 💻 Code: [GitHub link] · 🤗 Model: [HF link]
> Feedback from anyone working in agriculture is very welcome 🙏
> #AI #MachineLearning #AgriTech #ComputerVision #OpenSource

Also post: a 60–90 s screen recording (phone → photo → result → Hindi audio), r/developersIndia "I built" thread, Devpost, college tech fest.

## Demo video shot list (60–90 s)
1. (5 s) Hook: a diseased tomato leaf, "Is my crop sick?"
2. (10 s) Open app → switch to Hindi
3. (15 s) Take photo → result with confidence and severity
4. (15 s) "Get advice" → Hindi advice → press 🔊
5. (10 s) Ask "Can I use neem oil?" in chat
6. (10 s) Show a non-leaf photo being rejected (trustworthiness)
7. (10 s) Architecture slide + GitHub / Play Store links

## Real-farmer testing plan
1. Call your district **KVK**, explain it's a student project, ask for 30 minutes with a scientist + 5 farmers.
2. Ask each farmer to scan 3 leaves themselves. Note: did they understand the result? Did they use audio?
3. Ask the KVK scientist to review the advice for the 5 most common diseases in your area.
4. Record before/after changes in your report ("Farmers asked for bigger buttons → …").

## Viva / interview questions (prepare answers in your own words)
1. Why transfer learning instead of training from scratch?
2. Why freeze the backbone first, then fine-tune with a lower learning rate?
3. Why macro-F1 and not just accuracy? (class imbalance: 152 vs 5,507 images)
4. Why is PlantDoc accuracy much lower than PlantVillage? (domain shift: lab vs field backgrounds, lighting, multiple leaves)
5. What augmentations did you use and why do they mimic field conditions?
6. How does Grad-CAM work and what did it show for your model?
7. Why ONNX Runtime on the server instead of PyTorch? (size, speed, no GPU, portability)
8. What does INT8 quantization do and what is the accuracy trade-off?
9. How do you handle photos that are not leaves, or crops the model doesn't know?
10. What is RAG and why is it safer than asking the LLM directly?
11. How do you prevent the LLM from giving dangerous pesticide advice?
12. What happens if the LLM API is down or the key is missing?
13. How do you protect the API from abuse? (rate limiting, upload validation, size limits)
14. What personal data do you store? How does it relate to India's DPDP Act 2023?
15. Why a PWA + TWA instead of a native Android app?
16. Explain your CI/CD pipeline.
17. How would you improve accuracy on real field photos? (field data, feedback loop, PlantDoc fine-tuning, segmentation)
18. How would you scale to 1 lakh users? (stateless API, autoscaling, CDN for static files, managed DB, model caching)
19. What are the limitations and ethical risks of your system?
20. If you had 3 more months, what would you build next?

## Research paper ideas
1. **Lab-to-field robustness:** compare augmentation strategies / PlantDoc fine-tuning on the PlantVillage→PlantDoc domain gap.
2. **Grounded multilingual advisory:** measure hallucination rate of grounded vs ungrounded LLM advice, evaluated by KVK experts.
3. **Edge efficiency:** accuracy vs latency vs size of FP32/INT8 EfficientNet/MobileNet on low-end Android phones.
