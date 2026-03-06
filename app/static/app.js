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
const modelSelect = document.getElementById("model-select");
const imageInput = document.getElementById("image-input");
const imageUploadLabel = document.getElementById("image-upload-label");
const imageName = document.getElementById("image-name");
const galleryPanel = document.getElementById("gallery-panel");
const galleryList = document.getElementById("gallery-list");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");

let history = [];
let abortController = null;

// Lightbox: open on plot image click, close on click/Escape
function openLightbox(src) {
    lightboxImg.src = src;
    lightbox.hidden = false;
}

lightbox.addEventListener("click", () => { lightbox.hidden = true; });
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !lightbox.hidden) lightbox.hidden = true;
});

// Delegate click on plot images in messages
messagesDiv.addEventListener("click", (e) => {
    const img = e.target.closest("img[src^='/plots']");
    if (img) openLightbox(img.src);
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

function populateModels(models) {
    modelSelect.innerHTML = "";
    for (const model of models) {
        const opt = document.createElement("option");
        opt.value = model;
        opt.textContent = model.replace(/^models\//, "");
        modelSelect.appendChild(opt);
    }
    const preferred = models.find(m => m.includes("gemini-3.1-pro-preview"));
    if (preferred) modelSelect.value = preferred;
}

function addMessage(role, content, imageDataUrl) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    if (role === "assistant") {
        div.innerHTML = marked.parse(content);
        div.querySelectorAll("pre code").forEach(el => hljs.highlightElement(el));
    } else if (role === "code") {
        const pre = document.createElement("pre");
        const code = document.createElement("code");
        code.className = "language-python";
        code.textContent = content;
        hljs.highlightElement(code);
        pre.appendChild(code);
        div.appendChild(pre);
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
    imageUploadLabel.classList.remove("has-image");
    imageName.hidden = true;
    imageName.textContent = "";
}

// Track selected image
imageInput.addEventListener("change", () => {
    if (imageInput.files.length > 0) {
        imageUploadLabel.classList.add("has-image");
        imageName.textContent = imageInput.files[0].name;
        imageName.hidden = false;
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

// Check if already authenticated
fetch("/api/auth/status")
    .then(r => r.json())
    .then(data => {
        if (data.authenticated) {
            populateModels(data.models || []);
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
            populateModels(data.models || []);
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
    formData.append("model", modelSelect.value);
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
                } else if (eventType === "done") {
                    thinkingDiv.remove();
                    const data = JSON.parse(dataStr);
                    if (data.response) {
                        const userEntry = { role: "user", content: message };
                        if (data.image_info) {
                            userEntry.image = data.image_info;
                        }
                        history.push(userEntry);
                        history.push({ role: "assistant", content: data.response });
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
        sendBtn.textContent = "Send";
        sendBtn.classList.remove("stop");
        sendBtn.disabled = false;
        chatInput.focus();
    }
});

logoutBtn.addEventListener("click", async () => {
    await fetch("/api/auth/logout", { method: "POST" });
    history = [];
    messagesDiv.innerHTML = "";
    showLogin();
});
