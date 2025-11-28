document.getElementById("btn1").addEventListener("click", async function() {
  const text = document.getElementById("textInput").value.trim();
  const lang = document.getElementById("languageSelect").value;
  if (!text) return alert("Please enter some text first!");

  const res = await fetch("/speak", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, lang })
  });

  const data = await res.json();
  if (data.status === "success") {
    const audio = document.getElementById("audioPlayer");
    audio.src = data.file + "?t=" + new Date().getTime();
    audio.hidden = false;
    audio.play();
  } else {
    alert("Error: " + data.error);
  }
});
