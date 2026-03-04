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
        // Strip "models/" prefix for display
        opt.textContent = model.replace(/^models\//, "");
        modelSelect.appendChild(opt);
    }
    // Try to select a sensible default
    const preferred = models.find(m => m.includes("gemini-2.5-flash"));
    if (preferred) modelSelect.value = preferred;
}

function addMessage(role, content) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.textContent = content;
    messagesDiv.appendChild(div);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

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

    addMessage("user", message);
    chatInput.value = "";
    sendBtn.disabled = true;

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message, model: modelSelect.value, history }),
        });

        if (res.status === 401) {
            showLogin();
            return;
        }

        const data = await res.json();
        if (data.response) {
            history.push({ role: "user", content: message });
            history.push({ role: "assistant", content: data.response });
            addMessage("assistant", data.response);
        } else {
            addMessage("error", data.detail || "Something went wrong.");
        }
    } catch (err) {
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
