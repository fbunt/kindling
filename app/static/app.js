const loginView = document.getElementById("login-view");
const chatView = document.getElementById("chat-view");
const loginForm = document.getElementById("login-form");
const loginBtn = document.getElementById("login-btn");
const apiKeyInput = document.getElementById("api-key-input");
const loginError = document.getElementById("login-error");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const messagesDiv = document.getElementById("messages");
const logoutBtn = document.getElementById("logout-btn");
const clearBtn = document.getElementById("clear-btn");
const clearConfirmBtn = document.getElementById("clear-confirm-btn");
const MODEL = "gemini-3.1-pro-preview";
const imageInput = document.getElementById("image-input");
const imageUploadLabel = document.getElementById("image-upload-label");
const imageName = document.getElementById("image-name");
const galleryPanel = document.getElementById("gallery-panel");
const galleryList = document.getElementById("gallery-list");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");

let history = [];
let abortController = null;

// Lightbox: open on plot image click, toggle zoom, close on background/Escape
function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.hidden = false;
    lightbox.classList.remove("zoomed");
}

function closeLightbox() {
    lightbox.hidden = true;
    lightbox.classList.remove("zoomed");
}

lightboxImg.addEventListener("click", (e) => {
    e.stopPropagation();
    lightbox.classList.toggle("zoomed");
});
lightbox.addEventListener("click", (e) => {
    if (e.target === lightbox) closeLightbox();
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) closeLightbox();
});

// Delegate click on plot images in messages
messagesDiv.addEventListener("click", (e) => {
    const img = e.target.closest("img[src^='/plots']");
    if (img) openLightbox(img.src);
});

// marked's GFM strikethrough matches a single "~", so prose like
// "~20k pixels ... ~5k fires" renders struck-through. Require real "~~...~~".
// Returning undefined (not false) suppresses the match without falling back
// to marked's built-in single-tilde tokenizer.
marked.use({
    tokenizer: {
        del(src) {
            const match = /^~~(?=\S)([\s\S]*?\S)~~/.exec(src);
            if (!match) return undefined;
            return {
                type: "del",
                raw: match[0],
                text: match[1],
                tokens: this.lexer.inlineTokens(match[1]),
            };
        },
    },
});

function showLogin() {
    loginView.hidden = false;
    chatView.hidden = true;
}

function showChat() {
    loginView.hidden = true;
    chatView.hidden = false;
    chatInput.focus();
}


function addMessage(role, content, imageDataUrl) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    if (role === "assistant") {
        // Sanitize: model output incorporates web_search content, so rendered
        // markdown is untrusted — strip scripts/event handlers before insertion.
        div.innerHTML = DOMPurify.sanitize(marked.parse(content));
        div.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
        div.querySelectorAll("pre").forEach(pre => {
            pre.style.position = "relative";
            const copyBtn = document.createElement("button");
            copyBtn.className = "code-copy-btn";
            copyBtn.title = "Copy code";
            copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
            copyBtn.addEventListener("click", () => {
                const code = pre.querySelector("code");
                navigator.clipboard.writeText(code.innerText).then(() => {
                    copyBtn.classList.add("copied");
                    copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
                    setTimeout(() => {
                        copyBtn.classList.remove("copied");
                        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
                    }, 1500);
                });
            });
            pre.appendChild(copyBtn);
        });
    } else if (role === "code") {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Query";
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        code.className = "language-python";
        code.textContent = content;
        hljs.highlightElement(code);
        pre.appendChild(code);
        details.appendChild(summary);
        details.appendChild(pre);
        div.appendChild(details);
    } else {
        div.textContent = content;
    }
    if (imageDataUrl) {
        const img = document.createElement("img");
        img.src = imageDataUrl;
        div.appendChild(img);
    }
    // Add copy button for non-transient messages
    if (role !== "thinking") {
        const copyBtn = document.createElement("button");
        copyBtn.className = "copy-btn";
        copyBtn.title = "Copy to clipboard";
        copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
        copyBtn.addEventListener("click", () => {
            const text = role === "code" ? content : div.innerText;
            navigator.clipboard.writeText(text).then(() => {
                copyBtn.classList.add("copied");
                copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
                setTimeout(() => {
                    copyBtn.classList.remove("copied");
                    copyBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
                }, 1500);
            });
        });
        div.appendChild(copyBtn);
    }
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return div;
}

function addPlotToGallery(url, name) {
    galleryPanel.hidden = false;
    const item = document.createElement("div");
    item.className = "gallery-item";

    const img = document.createElement("img");
    img.src = url;
    img.alt = name;
    img.addEventListener("click", () => openLightbox(url));

    const label = document.createElement("div");
    label.className = "gallery-name";
    label.textContent = name;
    label.title = name;

    const attachBtn = document.createElement("button");
    attachBtn.className = "gallery-attach-btn";
    attachBtn.title = "Attach to message";
    attachBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 01-8.49-8.49l9.19-9.19a4 4 0 015.66 5.66l-9.2 9.19a2 2 0 01-2.83-2.83l8.49-8.48"/></svg>';
    attachBtn.addEventListener("click", async () => {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], name + ".png", { type: "image/png" });
        // Use DataTransfer to set the file on the input
        const dt = new DataTransfer();
        dt.items.add(file);
        imageInput.files = dt.files;
        imageInput.dispatchEvent(new Event("change"));
    });

    item.appendChild(img);
    item.appendChild(attachBtn);
    item.appendChild(label);
    galleryList.appendChild(item);
}

function clearImageInput() {
    imageInput.value = "";
    imageName.hidden = true;
    imageName.textContent = "";
    // Remove thumbnail card and restore upload button
    const card = document.getElementById("image-thumb-card");
    if (card) card.remove();
    imageUploadLabel.hidden = false;
}

// Track selected image — show thumbnail card replacing upload button
imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
        const file = imageInput.files[0];
        imageUploadLabel.hidden = true;
        imageName.hidden = true;

        // Create thumbnail card
        const card = document.createElement("div");
        card.className = "image-thumb-card";
        card.id = "image-thumb-card";

        const thumb = document.createElement("img");
        thumb.src = URL.createObjectURL(file);
        thumb.alt = file.name;

        const name = document.createElement("span");
        name.className = "image-thumb-name";
        name.textContent = file.name;
        name.title = file.name;

        const removeBtn = document.createElement("button");
        removeBtn.type = "button";
        removeBtn.className = "image-thumb-remove";
        removeBtn.title = "Remove image";
        removeBtn.innerHTML = "&times;";
        removeBtn.addEventListener("click", () => {
            clearImageInput();
        });

        card.appendChild(thumb);
        card.appendChild(name);
        card.appendChild(removeBtn);

        // Insert card where the upload label is
        imageUploadLabel.parentNode.insertBefore(card, imageUploadLabel);
    } else {
        clearImageInput();
    }
});

// Clear any stale file input from browser restore
clearImageInput();

// Auto-resize textarea
function autoResize() {
    chatInput.style.height = "auto";
    chatInput.style.height = chatInput.scrollHeight + "px";
}
chatInput.addEventListener("input", autoResize);

// Ctrl+Enter (or Cmd+Enter) to submit
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        chatForm.requestSubmit();
    }
});

// Check if already authenticated
fetch("/api/auth/status")
    .then(r => r.json())
    .then(data => {
        if (data.authenticated) {
            showChat();
        }
    });

loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    loginError.hidden = true;
    loginBtn.disabled = true;
    loginBtn.textContent = "Connecting...";

    try {
        const res = await fetch("/api/auth", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ api_key: apiKeyInput.value }),
        });
        const data = await res.json();
        if (data.ok) {
            showChat();
        } else {
            loginError.textContent = data.error;
            loginError.hidden = false;
        }
    } catch (err) {
        loginError.textContent = "Connection failed.";
        loginError.hidden = false;
    } finally {
        loginBtn.disabled = false;
        loginBtn.textContent = "Connect";
    }
});

sendBtn.addEventListener("click", (e) => {
    if (abortController) {
        e.preventDefault();
        abortController.abort();
    }
});

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = chatInput.value.trim();
    if (!message) return;

    // Read image if attached
    const imageFile = imageInput.files[0] || null;
    let imageDataUrl = null;
    if (imageFile) {
        imageDataUrl = await new Promise((resolve) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.readAsDataURL(imageFile);
        });
    }

    addMessage("user", message, imageDataUrl);
    chatInput.value = "";

    abortController = new AbortController();
    sendBtn.textContent = "Stop";
    sendBtn.classList.add("stop");

    const thinkingDiv = addMessage("thinking", "Thinking...");

    // Build form data
    const formData = new FormData();
    formData.append("message", message);
    formData.append("model", MODEL);
    formData.append("history", JSON.stringify(history));
    if (imageFile) {
        formData.append("image", imageFile);
    }
    clearImageInput();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            body: formData,
            signal: abortController.signal,
        });

        if (res.status === 401) {
            thinkingDiv.remove();
            showLogin();
            return;
        }

        // Parse SSE stream
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        const statusLabels = {
            thinking: "Thinking...",
            running_query: "Running query...",
        };

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            // Process complete SSE messages (separated by double newlines)
            let boundary;
            while ((boundary = buffer.indexOf("\n\n")) !== -1) {
                const raw = buffer.slice(0, boundary);
                buffer = buffer.slice(boundary + 2);

                let eventType = "message";
                let dataStr = "";
                for (const line of raw.split("\n")) {
                    if (line.startsWith("event: ")) eventType = line.slice(7);
                    else if (line.startsWith("data: ")) dataStr = line.slice(6);
                }
                if (!dataStr) continue;

                if (eventType === "status") {
                    const { status, queries } = JSON.parse(dataStr);
                    thinkingDiv.textContent = statusLabels[status] || status;
                    if (queries) {
                        for (const code of queries) {
                            const codeDiv = addMessage("code", code);
                            messagesDiv.insertBefore(codeDiv, thinkingDiv);
                        }
                    }
                } else if (eventType === "rejected") {
                    const { queries } = JSON.parse(dataStr);
                    for (const q of queries) {
                        const code = q.code;
                        const error = q.error || "Query failed";
                        // Find and mark the matching code block as rejected
                        const codeBlocks = messagesDiv.querySelectorAll(".message.code");
                        for (const block of codeBlocks) {
                            const codeEl = block.querySelector("code");
                            if (codeEl && codeEl.textContent === code) {
                                block.classList.add("rejected");
                                const label = document.createElement("span");
                                label.className = "code-error-label";
                                // Simplify display: strip "Query " prefix
                                label.textContent = error.replace(/^Query\s+/i, "");
                                const summary = block.querySelector("summary");
                                if (summary) {
                                    summary.appendChild(label);
                                } else {
                                    block.prepend(label);
                                }
                                break;
                            }
                        }
                    }
                } else if (eventType === "done") {
                    thinkingDiv.remove();
                    const data = JSON.parse(dataStr);
                    if (data.response) {
                        const userEntry = { role: "user", content: message };
                        if (data.image_info) {
                            userEntry.image = data.image_info;
                        }
                        history.push(userEntry);
                        const assistantEntry = { role: "assistant", content: data.response };
                        if (data.plot_images) {
                            assistantEntry.plot_images = data.plot_images;
                        }
                        history.push(assistantEntry);
                        addMessage("assistant", data.response);
                        if (data.plots) {
                            for (const plot of data.plots) {
                                addPlotToGallery(plot.url, plot.name);
                            }
                        }
                    } else {
                        addMessage("error", "Something went wrong.");
                    }
                } else if (eventType === "error") {
                    thinkingDiv.remove();
                    const { detail } = JSON.parse(dataStr);
                    addMessage("error", detail || "Something went wrong.");
                }
            }
        }
    } catch (err) {
        thinkingDiv.remove();
        if (err.name !== "AbortError") {
            addMessage("error", "Failed to send message.");
        }
    } finally {
        abortController = null;
        sendBtn.innerHTML = 'Send <kbd>Ctrl</kbd> <kbd>↵</kbd>';
        sendBtn.classList.remove("stop");
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

clearBtn.addEventListener("click", () => {
    const modal = new bootstrap.Modal(document.getElementById("clear-modal"));
    modal.show();
});

clearConfirmBtn.addEventListener("click", () => {
    if (abortController) abortController.abort();
    history = [];
    messagesDiv.innerHTML = "";
    galleryList.innerHTML = "";
    galleryPanel.hidden = true;
    clearImageInput();
    chatInput.focus();
    bootstrap.Modal.getInstance(document.getElementById("clear-modal")).hide();
});

logoutBtn.addEventListener("click", async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    history = [];
    messagesDiv.innerHTML = "";
    showLogin();
});

// Download chat as self-contained HTML
const downloadBtn = document.getElementById("download-btn");
downloadBtn.addEventListener("click", async () => {
    const messages = messagesDiv.querySelectorAll(".message");
    if (messages.length === 0) return;

    // Convert plot image URLs to base64 data URLs
    const imgPromises = [];
    const tempDiv = document.createElement("div");

    for (const msg of messages) {
        const clone = msg.cloneNode(true);
        // Remove copy buttons
        clone.querySelectorAll(".copy-btn").forEach(b => b.remove());
        // Expand collapsed details
        clone.querySelectorAll("details").forEach(d => d.open = true);
        tempDiv.appendChild(clone);
    }

    // Convert plot images to base64
    const plotImgs = tempDiv.querySelectorAll("img[src^='/plots']");
    for (const img of plotImgs) {
        imgPromises.push(
            fetch(img.src)
                .then(r => r.blob())
                .then(blob => new Promise((resolve) => {
                    const reader = new FileReader();
                    reader.onload = () => { img.src = reader.result; resolve(); };
                    reader.readAsDataURL(blob);
                }))
                .catch(() => {})
        );
    }
    await Promise.all(imgPromises);

    const timestamp = new Date().toISOString().slice(0, 19).replace(/[T:]/g, "-");
        const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>kindling chat - ${timestamp}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 2rem; }
.chat { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; gap: 0.75rem; }
h1 { text-align: center; color: #888; font-size: 1rem; margin-bottom: 1.5rem; }
.message { padding: 0.75rem 1rem; border-radius: 0.5rem; line-height: 1.5; max-width: 75%; word-wrap: break-word; }
.message.user { align-self: flex-end; background: #0d6efd; color: #fff; }
.message.assistant { align-self: flex-start; background: #2d2d3d; border: 1px solid #3d3d4d; }
.message.code { align-self: flex-start; background: #1e1e2e; border: 1px solid #3d3d4d; max-width: 85%; padding: 0; }
.message.code summary { padding: 0.4rem 0.75rem; font-size: 0.8rem; color: #888; cursor: pointer; }
.message.code pre { margin: 0; padding: 0.75rem 1rem; font-size: 0.82rem; border-top: 1px solid #3d3d4d; overflow-x: auto; }
.message.error { align-self: center; background: #3d1f1f; border: 1px solid #5a2d2d; color: #f5a5a5; }
.message img { max-width: 50%; border-radius: 0.25rem; margin-top: 0.5rem; cursor: zoom-in; }
.message table { width: 100%; border-collapse: collapse; margin: 0.5rem 0; font-size: 0.85rem; }
.message th, .message td { padding: 0.35rem 0.6rem; border: 1px solid #3d3d4d; text-align: left; }
.message th { background: #2a2a3a; font-weight: 600; }
.message pre { background: #2a2a3a; padding: 0.5rem 0.75rem; border-radius: 0.25rem; overflow-x: auto; }
.message code { font-size: 0.85em; }
.message p:last-child { margin-bottom: 0; }
.lightbox { position: fixed; inset: 0; z-index: 9999; background: rgba(0,0,0,0.85); display: none; align-items: center; justify-content: center; cursor: zoom-out; }
.lightbox.open { display: flex; }
.lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 0.5rem; box-shadow: 0 0 40px rgba(0,0,0,0.5); }
</style>
</head>
<body>
<h1>kindling chat &mdash; ${new Date().toLocaleString()}</h1>
<div class="chat">
${tempDiv.innerHTML}
</div>
<div class="lightbox" id="lb" onclick="this.classList.remove('open')"><img id="lb-img" src="" alt=""></div>
<script>
document.querySelectorAll('.message img').forEach(img => {
    img.addEventListener('click', () => {
        document.getElementById('lb-img').src = img.src;
        document.getElementById('lb').classList.add('open');
    });
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') document.getElementById('lb').classList.remove('open'); });
</script>
</body>
</html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `kindling-chat-${timestamp}.html`;
    a.click();
    URL.revokeObjectURL(url);
});
