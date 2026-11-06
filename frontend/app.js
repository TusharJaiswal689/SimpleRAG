const API_BASE = "";

const STORAGE_KEY = "college_notes_ai_chats";


/* =========================
   STATE
========================= */

let chats =
    JSON.parse(
        localStorage.getItem(STORAGE_KEY) || "[]"
    );

let activeChatId = null;

let generating = false;


/* =========================
   DOM
========================= */

const chatArea =
    document.getElementById("chatArea");

const chatHistory =
    document.getElementById("chatHistory");

const welcomeScreen =
    document.getElementById("welcomeScreen");

const chatForm =
    document.getElementById("chatForm");

const messageInput =
    document.getElementById("messageInput");

const sendBtn =
    document.getElementById("sendBtn");

const newChatBtn =
    document.getElementById("newChatBtn");

const fileInput =
    document.getElementById("fileInput");

const attachBtn =
    document.getElementById("attachBtn");

const selectedFile =
    document.getElementById("selectedFile");

const fileName =
    document.getElementById("fileName");

const fileStatus =
    document.getElementById("fileStatus");

const removeFile =
    document.getElementById("removeFile");

const uploadStatus =
    document.getElementById("uploadStatus");

const mobileMenu =
    document.getElementById("mobileMenu");

const sidebar =
    document.getElementById("sidebar");


/* =========================
   STORAGE
========================= */

function saveChats() {

    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(chats)
    );

}


/* =========================
   CHAT MANAGEMENT
========================= */

function createChat() {

    const chat = {

        id:
            crypto.randomUUID(),

        title:
            "New chat",

        messages:
            []

    };

    chats.unshift(chat);

    activeChatId =
        chat.id;

    saveChats();

    renderHistory();

    renderMessages();

}


function getActiveChat() {

    return chats.find(
        chat =>
            chat.id === activeChatId
    );

}


/* =========================
   HISTORY
========================= */

function renderHistory() {

    chatHistory.innerHTML = "";

    chats.forEach(chat => {

        const item =
            document.createElement("div");

        item.className =
            "history-item";

        if (
            chat.id === activeChatId
        ) {

            item.classList.add("active");

        }

        item.textContent =
            chat.title;

        item.onclick = () => {

            activeChatId =
                chat.id;

            renderHistory();

            renderMessages();

            sidebar.classList.remove("open");

        };

        chatHistory.appendChild(item);

    });

}


/* =========================
   MESSAGE DISPLAY
========================= */

function renderMessages() {

    const chat =
        getActiveChat();

    chatArea.innerHTML = "";

    if (
        !chat ||
        chat.messages.length === 0
    ) {

        showWelcome();

        return;

    }


    chat.messages.forEach(
        message => {

            addMessage(
                message.role,
                message.content,
                false,
                message.sources || []
            );

        }
    );


    scrollToBottom();

}


function showWelcome() {

    chatArea.innerHTML = `

        <div class="welcome-screen">

            <div class="welcome-icon">
                ✦
            </div>

            <h1>
                What can I help you learn?
            </h1>

            <p>
                Ask questions about your uploaded college notes.
            </p>

            <div class="suggestions">

                <button
                    onclick="useSuggestion('What is DBMS?')">

                    What is DBMS?

                </button>

                <button
                    onclick="useSuggestion('Explain the important concepts in my notes.')">

                    Explain important concepts

                </button>

                <button
                    onclick="useSuggestion('Summarize my notes.')">

                    Summarize my notes

                </button>

            </div>

        </div>

    `;

}


function escapeHtml(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\"/g, "&quot;")
        .replace(/'/g, "&#39;");

}

function renderAssistantHtml(content) {

    const text = String(content ?? "");

    if (typeof window === "undefined" || !window.marked || !window.DOMPurify) {
        return escapeHtml(text).replace(/\n/g, "<br>");
    }

    try {
        const parsed = window.marked.parse(text);
        return window.DOMPurify.sanitize(parsed, {
            ALLOWED_TAGS: [
                "p",
                "br",
                "strong",
                "b",
                "em",
                "i",
                "code",
                "pre",
                "ul",
                "ol",
                "li",
                "a",
                "blockquote",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "hr",
                "table",
                "thead",
                "tbody",
                "tr",
                "th",
                "td",
                "span",
                "div"
            ],
            ALLOWED_ATTR: [
                "href",
                "target",
                "rel",
                "class",
                "title"
            ],
            ALLOWED_URI_REGEXP: /^(?:(?:(?:f|ht)tps?|mailto):|[^a-z]|[a-z+.-]+(?:[^a-z+.-:]|$))/i
        });
    }
    catch (error) {
        return escapeHtml(text).replace(/\n/g, "<br>");
    }

}

function addMessage(
    role,
    content,
    streaming = false,
    sources = []
) {

    const message =
        document.createElement("div");

    message.className =
        `message ${role}`;


    const avatar =
        document.createElement("div");

    avatar.className =
        "avatar";

    avatar.textContent =
        role === "user"
            ? "U"
            : "✦";


    const contentElement =
        document.createElement("div");

    contentElement.className =
        "message-content";

    if (role === "assistant") {

        contentElement.innerHTML =
            renderAssistantHtml(content);

    } else {

        contentElement.textContent =
            content;

    }


    if (streaming) {

        contentElement.innerHTML =
            renderAssistantHtml(content) +
            `<span class="streaming-cursor"></span>`;

    }

    if (role === "assistant" && sources.length > 0) {
        renderSources(contentElement, sources);
    }

    message.appendChild(avatar);

    message.appendChild(contentElement);

    chatArea.appendChild(message);


    return contentElement;

}

function renderSources(contentElement, sources) {

    const visibleSources = sources.filter(
        source => source.filename || source.page_number !== undefined
    );
    if (visibleSources.length === 0) {
        return;
    }

    const sourceList = document.createElement("div");
    sourceList.className = "retrieved-sources";

    const heading = document.createElement("strong");
    heading.textContent = "Retrieved sources (not claim-verified citations)";
    sourceList.appendChild(heading);

    const list = document.createElement("ul");
    visibleSources.forEach(source => {
        const item = document.createElement("li");
        const parts = [];
        if (source.filename) {
            parts.push(source.filename);
        }
        if (source.page_number !== undefined) {
            parts.push(`page ${source.page_number}`);
        }
        item.textContent = parts.join(", ");
        list.appendChild(item);
    });

    sourceList.appendChild(list);
    contentElement.appendChild(sourceList);
}


/* =========================
   STREAMING
========================= */

async function sendMessage(query) {

    if (
        !query.trim() ||
        generating
    ) {

        return;

    }


    if (!activeChatId) {

        createChat();

    }


    const chat =
        getActiveChat();


    /*
       Save user message
    */

    chat.messages.push({

        role:
            "user",

        content:
            query

    });


    /*
       Give chat its title
    */

    if (
        chat.title === "New chat"
    ) {

        chat.title =
            query.length > 35
                ? query.substring(0, 35) + "..."
                : query;

    }


    saveChats();

    renderHistory();


    /*
       Clear welcome / redraw
    */

    chatArea.innerHTML = "";


    /*
       Render existing messages
    */

    chat.messages.forEach(
        message => {

            addMessage(
                message.role,
                message.content
            );

        }
    );


    /*
       Create empty assistant message
    */

    const assistantElement =
        addMessage(
            "assistant",
            "",
            true
        );


    scrollToBottom();


    generating = true;

    sendBtn.disabled = true;

    messageInput.disabled = true;


    try {

        /*
           Send previous conversation
           as history.

           We exclude the current
           question because it is
           already sent separately.
        */

        const history =
            chat.messages
                .slice(0, -1);


        const response =
            await fetch(
                `${API_BASE}/chat/`,
                {

                    method:
                        "POST",

                    headers: {

                        "Content-Type":
                            "application/json",

                        "Accept":
                            "text/event-stream"

                    },

                    body:
                        JSON.stringify({

                            query:
                                query,

                            history:
                                history

                        })

                }
            );


        if (!response.ok) {

            const error =
                await response.text();

            throw new Error(
                error || "Chat request failed"
            );

        }


        if (!response.body) {

            throw new Error(
                "Streaming response is unavailable."
            );

        }


        /*
           THIS IS THE IMPORTANT PART.

           We do NOT use:

           await response.text()

           because that would wait
           for the complete response.

           Instead we read chunks
           as they arrive.
        */

        const reader =
            response.body.getReader();


        const decoder =
            new TextDecoder();

        let answer = "";
        let sources = [];
        let eventBuffer = "";

        function consumeEvent(frame) {
            if (!frame.trim()) {
                return;
            }

            let eventName = "message";
            const dataLines = [];
            frame.split(/\r?\n/).forEach(line => {
                if (line.startsWith("event:")) {
                    eventName = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                    dataLines.push(line.slice(5).trimStart());
                }
            });

            if (dataLines.length === 0) {
                return;
            }

            const data = JSON.parse(dataLines.join("\n"));
            if (eventName === "token") {
                answer += data;
                assistantElement.innerHTML =
                    renderAssistantHtml(answer) +
                    `<span class="streaming-cursor"></span>`;
                scrollToBottom();
            } else if (eventName === "sources" && Array.isArray(data)) {
                sources = data;
            }
        }


        while (true) {

            const {
                value,
                done
            } =
                await reader.read();


            if (done) {

                break;

            }


            eventBuffer += decoder.decode(value, {stream: true});
            const frames = eventBuffer.split(/\r?\n\r?\n/);
            eventBuffer = frames.pop() || "";
            frames.forEach(consumeEvent);

        }

        eventBuffer += decoder.decode();
        if (eventBuffer.trim()) {
            consumeEvent(eventBuffer);
        }

        /*
           Remove cursor
        */

        assistantElement.innerHTML =
            renderAssistantHtml(answer);
        renderSources(assistantElement, sources);


        /*
           Save assistant response
        */

        chat.messages.push({

            role:
                "assistant",

            content:
                answer,

            sources:
                sources

        });


        saveChats();

    }

    catch (error) {

        assistantElement.textContent =
            `Error: ${error.message}`;

    }

    finally {

        generating = false;

        sendBtn.disabled = false;

        messageInput.disabled = false;

        messageInput.focus();

    }

}


/* =========================
   FORM
========================= */

chatForm.addEventListener(
    "submit",
    async event => {

        event.preventDefault();


        const query =
            messageInput.value.trim();


        if (!query) {

            return;

        }


        messageInput.value = "";

        messageInput.style.height =
            "auto";


        await sendMessage(query);

    }
);


/* =========================
   TEXTAREA
========================= */

messageInput.addEventListener(
    "input",
    () => {

        messageInput.style.height =
            "auto";


        messageInput.style.height =
            Math.min(
                messageInput.scrollHeight,
                180
            ) + "px";

    }
);


messageInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            chatForm.requestSubmit();

        }

    }
);


/* =========================
   FILE UPLOAD
========================= */

attachBtn.addEventListener(
    "click",
    () => {

        fileInput.click();

    }
);


fileInput.addEventListener(
    "change",
    async () => {

        const file =
            fileInput.files[0];


        if (!file) {

            return;

        }


        const validTypes =
            [
                ".pdf",
                ".docx"
            ];


        const extension =
            "." +
            file.name
                .split(".")
                .pop()
                .toLowerCase();


        if (
            !validTypes.includes(extension)
        ) {

            alert(
                "Please select a PDF or DOCX file."
            );

            fileInput.value = "";

            return;

        }


        selectedFile.classList.remove(
            "hidden"
        );


        fileName.textContent =
            file.name;


        fileStatus.textContent =
            "Uploading...";


        uploadStatus.textContent =
            `Uploading ${file.name}...`;


        try {

            const formData =
                new FormData();


            formData.append(
                "file",
                file
            );


            const response =
                await fetch(
                    `${API_BASE}/documents/upload`,
                    {

                        method:
                            "POST",

                        body:
                            formData

                    }
                );


            if (!response.ok) {

                const error =
                    await response.text();

                throw new Error(
                    error ||
                    "Upload failed"
                );

            }


            const result =
                await response.json();


            fileStatus.textContent =
                `${result.chunks_indexed} chunks indexed`;


            uploadStatus.textContent =
                `✓ ${result.filename} indexed successfully`;


        }

        catch (error) {

            fileStatus.textContent =
                "Upload failed";


            uploadStatus.textContent =
                `Upload error: ${error.message}`;

        }

    }
);


/* =========================
   REMOVE FILE
========================= */

removeFile.addEventListener(
    "click",
    () => {

        fileInput.value = "";

        selectedFile.classList.add(
            "hidden"
        );

        uploadStatus.textContent =
            "";

    }
);


/* =========================
   NEW CHAT
========================= */

newChatBtn.addEventListener(
    "click",
    () => {

        createChat();

        messageInput.focus();

    }
);


/* =========================
   SUGGESTIONS
========================= */

function useSuggestion(text) {

    messageInput.value =
        text;

    messageInput.focus();

    messageInput.style.height =
        "auto";

    messageInput.style.height =
        Math.min(
            messageInput.scrollHeight,
            180
        ) + "px";

}


/* =========================
   MOBILE MENU
========================= */

mobileMenu.addEventListener(
    "click",
    () => {

        sidebar.classList.toggle(
            "open"
        );

    }
);


/* =========================
   SCROLL
========================= */

function scrollToBottom() {

    chatArea.scrollTop =
        chatArea.scrollHeight;

}


/* =========================
   HTML ESCAPING
========================= */

function escapeHTML(text) {

    const div =
        document.createElement("div");

    div.textContent =
        text;

    return div.innerHTML;

}


/* =========================
   INITIALIZATION
========================= */

if (chats.length === 0) {

    createChat();

}

else {

    activeChatId =
        chats[0].id;

    renderHistory();

    renderMessages();

}