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

let history = [];

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
    div.textContent = content;
    if (imageDataUrl) {
        const img = document.createElement("img");
        img.src = imageDataUrl;
        div.appendChild(img);
    }
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    return div;
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
    sendBtn.disabled = true;

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
        });

        thinkingDiv.remove();

        if (res.status === 401) {
            showLogin();
            return;
        }

        const data = await res.json();
        if (data.response) {
            const userEntry = { role: "user", content: message };
            if (data.image_info) {
                userEntry.image = data.image_info;
            }
            history.push(userEntry);
            history.push({ role: "assistant", content: data.response });
            addMessage("assistant", data.response);
        } else {
            addMessage("error", data.detail || "Something went wrong.");
        }
    } catch (err) {
        thinkingDiv.remove();
        addMessage("error", "Failed to send message.");
    } finally {
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
