const emojies = document.querySelectorAll(".emojies");
const container = document.querySelector(".container");

function deleteContent() {
  container.innerHTML = "";
}
function feedback(value) {
  console.log(value);
}

emojies.forEach((emoji) => {
  emoji.addEventListener("click", () => {
    deleteContent();
    feedback(emoji.dataset.value);
  });
});
