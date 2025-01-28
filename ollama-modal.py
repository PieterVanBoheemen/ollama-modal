import modal
import os
import subprocess
import time

from modal import build, enter, method

MODEL = os.environ.get("MODEL", "llama3.3")

def pull(model: str = MODEL):
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(2)  # 2s, wait for the service to start
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)

image = (
    modal.Image
    .debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands( # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .copy_local_file("ollama.service", "/etc/systemd/system/ollama.service")
    .pip_install("ollama") #, force_build=True) # add force_build=True when ever you change the model
    .run_function(pull)
)

app = modal.App(name="ollama", image=image)

with image.imports():
    import ollama

@app.cls(gpu='h100:4', container_idle_timeout=300)
class Ollama:
    @build()
    def pull(self):
        ...

    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])
        time.sleep(2) # wait for it

    @method()
    def infer(self, text: str):
        # Configure ollama to use all available GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(4))

        stream = ollama.chat(
            model=MODEL,
            messages=[{'role': 'user', 'content': text}],
            stream=True,
            options={"num_gpu": 4}  # Specify number of GPUs to use
        )
        for chunk in stream:
            yield chunk['message']['content']
            print(chunk['message']['content'], end='', flush=True)
        return

@app.local_entrypoint()
def main(text: str = "Why is the sky blue?", lookup: bool = False):
    if lookup:
        ollama = modal.Cls.lookup("ollama", "Ollama")
    else:
        ollama = Ollama()
    for chunk in ollama.infer.remote_gen(text):
        print(chunk, end='', flush=False)
