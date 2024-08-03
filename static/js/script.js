const socket = io();

document.getElementById("start-record").onclick = function () {
  this.disabled = true;
  document.getElementById("stop-record").disabled = false;
  startRecording();
};

document.getElementById("stop-record").onclick = function () {
  this.disabled = true;
  document.getElementById("start-record").disabled = false;
  stopRecording();
};

let mediaRecorder;
let audioChunks = [];
let sendInterval;

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.start();

  mediaRecorder.addEventListener("dataavailable", (event) => {
    audioChunks.push(event.data);
  });

  sendInterval = setInterval(() => {
    if (audioChunks.length > 0) {
      let combinedBlob = new Blob(audioChunks);
      if (combinedBlob.size >= 4096) {
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64String = reader.result.split(",")[1];
          sendData(base64String);
          audioChunks = []; // Reset audioChunks sau khi gửi
        };
        reader.readAsDataURL(combinedBlob.slice(0, 4096)); // Lấy đúng 4096 bytes
        audioChunks = [combinedBlob.slice(4096)]; // Lưu lại phần còn dư
      }
    }
  }, 35);
}

function stopRecording() {
  clearInterval(sendInterval);
  mediaRecorder.stop();
}

function sendData(base64String) {
  $.ajax({
    type: "POST",
    url: "/send_audio",
    data: { audio_data: base64String },
    success: function (response) {
      console.log("Data sent successfully");
    },
    error: function (error) {
      console.error("Error sending data:", error);
    },
  });
}

// Kết nối WebSocket
socket.on("connect", function () {
  console.log("Connected to WebSocket");
});

socket.on("disconnect", function () {
  console.log("Disconnected from WebSocket");
});

// Xử lý khi nhận được thông điệp từ server qua WebSocket
socket.on("message", function (data) {
  console.log("Received message from server:", data);
});
