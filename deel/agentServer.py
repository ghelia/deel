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
	log_file = 'log_reward.log'
	reward_sum = 0

	def received_message(self, m):
		print ("received")
		payload = m.data

		dat = msgpack.unpackb(payload)
		screen = Image.open(io.BytesIO(bytearray(dat['image'])))
		x = screen
		AgentServer.reward = dat['reward']
		end_episode = dat['endEpisode']

		if not self.agent_initialized:
			self.agent_initialized = True
			self.agent.agent_init(Deel.gpu)

			AgentServer.mode='start'
			#action = self.agent.agent_start(image)
			action = workout(x)
			self.send(str(action))
			with open(self.log_file, 'w') as the_file:
				the_file.write('cycle, episode_reward_sum \n')			
		else:
			self.thread_event.wait()
			self.cycle_counter += 1
			self.reward_sum += AgentServer.reward

			if end_episode:
				AgentServer.mode='end'
				workout(x)
				#self.agent.agent_end(reward)
				AgentServer.mode='start'
				#action = self.agent.agent_start(image)  # TODO
				action = workout(x)
				self.send(str(action))
				with open(self.log_file, 'a') as the_file:
					the_file.write(str(self.cycle_counter) +
								   ',' + str(self.reward_sum) + '\n')
				self.reward_sum = 0

			else:
				#action, rl_action, eps, Q_now, obs_array, returnAction = self.agent.agent_step(reward, image)
				#self.agent.agent_step_after(reward, image, rl_action, eps, Q_now, obs_array, returnAction)
				AgentServer.mode='step'
				ag,action, eps, Q_now, obs_array = workout(x)
				self.send(str(action))
				ag.step_after(AgentServer.reward, action, eps, Q_now, obs_array)

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

