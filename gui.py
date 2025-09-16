import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import subprocess
import threading

class SignLanguageGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sign Language Recognition")
        self.geometry("700x550")

        label = tk.Label(self, text="Type the word you want to train for and click Collect Data:", font=("Arial", 12))
        label.pack(pady=5)

        input_frame = tk.Frame(self)
        input_frame.pack(pady=5)

        self.action_word_var = tk.StringVar()
        self.action_entry = tk.Entry(input_frame, textvariable=self.action_word_var, width=30, font=("Arial", 14))
        self.action_entry.grid(row=0, column=0, padx=10)

        self.btn_collect = tk.Button(input_frame, text="Collect Data", width=20, command=self.run_collect)
        self.btn_collect.grid(row=0, column=1)

        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)

        self.btn_train = tk.Button(btn_frame, text="Train Model", width=20, command=self.run_train)
        self.btn_predict = tk.Button(btn_frame, text="Predict Live", width=20, command=self.run_predict)

        self.btn_train.grid(row=0, column=0, padx=10)
        self.btn_predict.grid(row=0, column=1, padx=10)

        self.output_text = ScrolledText(self, height=20, width=80, state='disabled', font=("Consolas", 10))
        self.output_text.pack(pady=10)

        self.process_thread = None

    def run_collect(self):
        word = self.action_word_var.get().strip()
        if not word:
            messagebox.showwarning("Warning", "Please type the action word before collecting data.")
            return
        self.run_script(['python3', 'data_collection_script.py', word])

    def run_train(self):
        self.run_script(['python3', 'train_model_script.py'])

    def run_predict(self):
        self.run_script(['python3', 'live_prediction_script.py'])

    def run_script(self, cmd):
        if self.process_thread is not None and self.process_thread.is_alive():
            messagebox.showwarning("Warning", "A process is already running.")
            return

        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')

        def target():
            try:
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    self.append_output(line)
                process.wait()
                self.append_output("\nProcess finished.")
            except Exception as e:
                self.append_output(f"Error: {e}")

        self.process_thread = threading.Thread(target=target)
        self.process_thread.start()

    def append_output(self, text):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)
        self.output_text.config(state='disabled')

if __name__ == "__main__":
    app = SignLanguageGUI()
    app.mainloop()
