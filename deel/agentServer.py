import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from agent import *
import msgpack
import io
from PIL import Image
import threading


class Root(object):
	@cherrypy.expose
	def index(self):
		return 'some HTML with a websocket javascript connection'

	@cherrypy.expose
	def ws(self):
		# you can access the class instance through
		handler = cherrypy.request.ws_handler

workout= None
class AgentServer(WebSocket):
	agent = Agent()
	agent_initialized = False
	cycle_counter = 0
	thread_event = threading.Event()
	trainer = None
	mode='none'
	reward=None

	def received_message(self, m):
		print ("received")
		payload = m.data

		dat = msgpack.unpackb(payload)
		screen = Image.open(io.BytesIO(bytearray(dat['image'])))
		x = screen
		print x
		print type(x)
		'''
		print screen
		x = ImageTensor(screen,filtered_image=np.asarray(screen).transpose(2, 0, 1)[::-1])
		x.use()
		'''
		AgentServer.reward = dat['reward']
		end_episode = dat['endEpisode']
		# image.save(str(self.counter) + ".png")

		if not self.agent_initialized:
			self.agent_initialized = True
			self.agent.agent_init(Deel.gpu)

			AgentServer.mode='start'
			#action = self.agent.agent_start(image)
			action = workout(x)
		else:
			self.thread_event.wait()
			self.cycle_counter += 1

			if end_episode:
				AgentServer.mode='end'
				workout(x)
				#self.agent.agent_end(reward)
				AgentServer.mode='start'
				#action = self.agent.agent_start(image)  # TODO
				action = workout(x)

			else:
				#action, rl_action, eps, Q_now, obs_array, returnAction = self.agent.agent_step(reward, image)
				#self.agent.agent_step_after(reward, image, rl_action, eps, Q_now, obs_array, returnAction)
				AgentServer.mode='step'
				action = workout(x)

		print str(action.intArray[0])
		self.send(str(action.intArray[0]))
		self.thread_event.set()

def StartAgent(trainer=None,port=8765):
	global workout
	workout = trainer
	cherrypy.config.update({'server.socket_port': port})
	WebSocketPlugin(cherrypy.engine).subscribe()
	cherrypy.tools.websocket = WebSocketTool()
	cherrypy.config.update({'engine.autoreload.on': False})
	config = {'/ws': {'tools.websocket.on': True,
					  'tools.websocket.handler_cls': AgentServer}}
	cherrypy.quickstart(Root(), '/', config)

