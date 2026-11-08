"""Offline AI answers with BLIP (free, no API key, no server overload).
First use downloads about 2 GB of model files (internet needed once)."""
_cache = {}

DESCRIBE_WORDS = ("describe", "what is happening", "what's happening", "caption",
                  "tell me about", "explain", "what is in", "what is this", "what do you see")


def _load(kind):
    if kind not in _cache:
        from transformers import (BlipForConditionalGeneration, BlipForQuestionAnswering,
                                  BlipProcessor)
        if kind == "caption":
            name, cls = "Salesforce/blip-image-captioning-base", BlipForConditionalGeneration
        else:
            name, cls = "Salesforce/blip-vqa-base", BlipForQuestionAnswering
        _cache[kind] = (BlipProcessor.from_pretrained(name), cls.from_pretrained(name))
    return _cache[kind]


def answer(pil_image, question):
    import torch
    img = pil_image.convert("RGB")
    if any(w in question.lower() for w in DESCRIBE_WORDS):
        proc, model = _load("caption")
        inputs = proc(img, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=60, num_beams=3)
        text = proc.decode(out[0], skip_special_tokens=True).strip()
        return (text[:1].upper() + text[1:] + ".") if text else "I could not describe this image."
    proc, model = _load("vqa")
    inputs = proc(img, question, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return proc.decode(out[0], skip_special_tokens=True).strip()
