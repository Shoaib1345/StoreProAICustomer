        setTimeout(() => {
            document.getElementById('introScreen').classList.add('fade-out');
            setTimeout(() => {
                document.getElementById('introScreen').style.display = 'none';
                document.getElementById('chatInput').focus();
            }, 800);
        }, 2500);

        function quickAction(text) {
            document.getElementById('chatInput').value = text;
            sendMessage();
        }

        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }

        async function sendMessage() {
            const input = document.getElementById('chatInput');
            const text = input.value.trim();

            if (!text) return;

            addMessage(text, 'user');
            input.value = '';

            const typingIndicator = document.getElementById('typingIndicator');
            typingIndicator.style.display = 'block';
            scrollToBottom();

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: text })
                });

                const data = await response.json();
                typingIndicator.style.display = 'none';
                addMessage(data.answer, 'bot');

            } catch (error) {
                typingIndicator.style.display = 'none';
                addMessage('⚠️ Connection error. Please try again.', 'bot');
            }
        }

        function addMessage(text, sender) {
            const container = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = sender === 'user' ? 'message user-msg' : 'message bot-msg';
            messageDiv.textContent = text;
            container.appendChild(messageDiv);
            scrollToBottom();
        }

        function scrollToBottom() {
            const container = document.getElementById('chatMessages');
            container.scrollTop = container.scrollHeight;
        }