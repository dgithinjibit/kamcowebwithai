document.addEventListener('DOMContentLoaded', function () {
    const chatbotBtn = document.getElementById('chatbotBtn');
    const chatbotPanel = document.getElementById('chatbotPanel');
    const chatbotClose = document.getElementById('chatbotClose');
    const chatbotInput = document.getElementById('chatbotInput');
    const chatbotSend = document.getElementById('chatbotSend');
    const chatbotMessages = document.getElementById('chatbotMessages');

    // Show welcome message after 0.5s
    setTimeout(() => {
        const typing = chatbotMessages.querySelector('.typing-indicator')?.parentElement;
        if (typing) typing.remove();

        addMessage("Welcome! I'm the Kamwenja TTC assistant. Ask me about fees, courses, or admission requirements.", 'bot');
    }, 500);

    // Toggle chat
    chatbotBtn.addEventListener('click', () => {
        chatbotPanel.style.display = chatbotPanel.style.display === 'block' ? 'none' : 'block';
        if (chatbotPanel.style.display === 'block') chatbotInput.focus();
    });

    chatbotClose.addEventListener('click', () => {
        chatbotPanel.style.display = 'none';
    });

    // Send message
    function sendMessage() {
        const question = chatbotInput.value.trim();
        if (!question) return;

        addMessage(question, 'user');
        chatbotInput.value = '';

        // Show typing effect
        const typing = document.createElement('div');
        typing.className = 'chatbot-message bot';
        typing.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
        chatbotMessages.appendChild(typing);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;

        // Fast response after 0.6s
        setTimeout(() => {
            typing.remove();
            const answer = getResponse(question);
            addMessage(answer, 'bot');
        }, 600);
    }

    function addMessage(text, sender) {
        const msg = document.createElement('div');
        msg.className = `chatbot-message ${sender}`;
        msg.textContent = text;
        chatbotMessages.appendChild(msg);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    // Smart Response Logic
    function getResponse(input) {
        const lower = input.toLowerCase();

        const fees = "First Year: KES 55,000 per term\nSecond Year: KES 50,000\nThird Year: KES 48,000\nBoarding (Optional): KES 15,000\nPayment: KCB Bank (Acct: 123456789) or M-Pesa Paybill 987654";

        const courses = "We offer:\n• Diploma in Primary Teacher Education (3 years)\n• Diploma in Early Childhood Education (3 years)\n• Certificate to Diploma Upgrade (1 year)";

        const admission = "Admission Requirements:\n• KCSE Certificate (Min C Plain)\n• National ID or Birth Certificate\n• 4 Passport Photos\n• Application Fee: KES 1,000\nDeadline: July 31st";

        const location = "Kamwenja TTC is located in Nyeri County, Kenya.";

        const dates = "Important Dates:\n• Application Deadline: July 31\n• Admission Letters: August 15\n• Reporting Date: September 5";

        if (lower.includes('fee') || lower.includes('cost') || lower.includes('payment')) return fees;
        if (lower.includes('course') || lower.includes('program') || lower.includes('study')) return courses;
        if (lower.includes('admission') || lower.includes('apply') || lower.includes('requirement')) return admission;
        if (lower.includes('location') || lower.includes('where')) return location;
        if (lower.includes('date') || lower.includes('deadline') || lower.includes('when')) return dates;
        if (lower.includes('hello') || lower.includes('hi')) return "Hello! How can I help you today?";
        if (lower.includes('thank')) return "You're welcome! Need help with anything else?";
        
        return "I don't have that info. Please contact the admissions office at +254 726 238 121.";
    }

    // Event Listeners
    chatbotSend.addEventListener('click', sendMessage);
    chatbotInput.addEventListener('keypress', e => {
        if (e.key === 'Enter') sendMessage();
    });

    // Close chat when clicking outside
    document.addEventListener('click', e => {
        if (!chatbotPanel.contains(e.target) && 
            e.target !== chatbotBtn && 
            !chatbotBtn.contains(e.target)) {
            chatbotPanel.style.display = 'none';
        }
    });
});
