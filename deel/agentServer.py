import cherrypy
import argparse
from ws4py.server.cherrypyserver import WebSocketPlugin, WebSocketTool
from ws4py.websocket import WebSocket
from agent import *
import msgpack
import io
from PIL import Image
import threading


parser = argparse.ArgumentParser(description='UnitySocketController')
parser.add_argument('--port', '-p', default='8765', type=int,
                    help='websocket port')
parser.add_argument('--ip', '-i', default='127.0.0.1',
                    help='server ip')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()


class Root(object):
    @cherrypy.expose
    def index(self):
        return 'some HTML with a websocket javascript connection'

    @cherrypy.expose
    def ws(self):
        # you can access the class instance through
        handler = cherrypy.request.ws_handler


class AgentServer(WebSocket):
    agent = Agent()
    agent_initialized = False
    cycle_counter = 0
    thread_event = threading.Event()
    trainer = None

    def received_message(self, m):
        print ("received")
        payload = m.data

        dat = msgpack.unpackb(payload)
        image = Image.open(io.BytesIO(bytearray(dat['image'])))
        reward = dat['reward']
        end_episode = dat['endEpisode']
        # image.save(str(self.counter) + ".png")

        if not self.agent_initialized:
            self.agent_initialized = True
            self.agent.agent_init(args.gpu)
            action = self.agent.agent_start(image)
        else:
            self.thread_event.wait()
            self.cycle_counter += 1

            if end_episode:
                self.agent.agent_end(reward)
                action = self.agent.agent_start(image)  # TODO
            else:
                action, rl_action, eps, Q_now, obs_array, returnAction = self.agent.agent_step(reward, image)
                self.agent.agent_step_after(reward, image, rl_action, eps, Q_now, obs_array, returnAction)

        print str(action.intArray[0])
        self.send(str(action.intArray[0]))
        self.thread_event.set()

def StartAgent(trainer=None,port=8765):
    AgentServer.trainer = trainer
    cherrypy.config.update({'server.socket_port': port})
    WebSocketPlugin(cherrypy.engine).subscribe()
    cherrypy.tools.websocket = WebSocketTool()
    cherrypy.config.update({'engine.autoreload.on': False})
    config = {'/ws': {'tools.websocket.on': True,
                      'tools.websocket.handler_cls': AgentServer}}
    cherrypy.quickstart(Root(), '/', config)

