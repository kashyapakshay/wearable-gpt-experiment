import base64
import os
import socket
import struct
import threading
import cv2
import gradio as gr
import numpy as np
from io import BytesIO
from openai import OpenAI


OPENAI_KEY = os.getenv('OPENAI_KEY')
assert OPENAI_KEY is not None
client = OpenAI(api_key=OPENAI_KEY)

HOST = os.getenv('WEAR_HOST', '0.0.0.0')
PORT = 5190

class Pipeline():
	def __init__(self):
		self.message = None
		self._lock = threading.Lock()

	def get_message(self):
		self._lock.acquire()
		message = self.message
		self._lock.release()
		return message

	def set_message(self, value):
		self._lock.acquire()
		self.message = value
		self._lock.release()

def frames_streaming(pipeline):
	with socket.socket() as s:
		s.bind((HOST, PORT))
		s.listen(0)
		conn, addr = s.accept()
		connection = conn.makefile('rb')
		fc = 0

		try:
			while True:
				image_len = struct.unpack('<L', connection.read(struct.calcsize('<L')))[0]
				if not image_len:
					break
				image_stream = BytesIO()
				image_stream.write(connection.read(image_len))
				image_stream.seek(0)
				image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), cv2.IMREAD_COLOR)
				image = image[::-1,::-1,:]
				cv2.imwrite('current_frame.jpg', image)

				fc += 1

				if fc % 30 == 0:
					_, buffer = cv2.imencode(".jpg", image)
					encoded = base64.b64encode(buffer).decode("utf-8")
					pipeline.set_message(encoded)
		finally:
			connection.close()
			conn.close()
			cv2.destroyAllWindows()
			pipeline.set_message(None)

def _respond(pipeline, prompt, chat_history):
	encoded = pipeline.get_message()

	PROMPT_MESSAGES = [
		{
			"role": "user",
			"content": [
				prompt,
				{'image' : encoded, 'resize': 768},
			],
		},
	]
	params = {
		"model": "gpt-4-vision-preview",
		"messages": PROMPT_MESSAGES,
		"max_tokens": 100,
	}
	result = client.chat.completions.create(**params)
	response = result.choices[0].message.content
	chat_history.append((prompt, response))
	return "", chat_history

if __name__ == '__main__':
	from functools import partial

	img_pipeline = Pipeline()
	threads = [
		threading.Thread(target=frames_streaming, args=(img_pipeline,)),
	]
	print('Starting threads...')
	[x.start() for x in threads]

	with gr.Blocks() as demo:
		gr.Markdown("# SecondBrain 🧠")
		with gr.Column():
			chatbot = gr.Chatbot(height=500, bubble_full_width=False)
			message_textbox = gr.Textbox()

		message_textbox.submit(
			fn=partial(_respond, img_pipeline),
			inputs=[message_textbox, chatbot],
			outputs=[message_textbox, chatbot]
		)
	demo.launch(debug=False, show_error=True)

	print('Joining threads...')
	[x.join() for x in threads]
