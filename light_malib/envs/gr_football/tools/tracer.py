# MIT License

# Copyright (c) 2022 DigitalBrain, Yan Song and He jiang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pickle
from PIL import Image
import io

class MatchTracer:
    '''
    Tracer only used for data storage.
    '''
    def __init__(self,no_frame=False) -> None:
        self.data={}
        self.step=-1
        self.curr=None
        self.no_frame=no_frame
        self.extra_data={}
    
    def inc_step(self):
        self.step+=1
        self.data[self.step]={}
        self.curr=self.data[self.step]
        
    def update_settings(self,settings):
        self.n_left_control=settings["n_left_control"]
        self.n_right_control=settings["n_right_control"]
    
    def update(self,observations,actions=None):
        self.inc_step()
        controls={"left":{},"right":{}}
        for i,obs in enumerate(observations):
            if i==0:
                self.curr.update(obs)
                if "active" in self.curr:
                    self.curr.pop("active")
                if "sticky_actions" in self.curr:
                    self.curr.pop("sticky_actions")
                if self.step!=0:
                    self.curr.pop("left_team_roles")
                    self.curr.pop("right_team_roles")
                else:
                    self.curr["n_left_control"]=self.n_left_control
                    self.curr["n_right_control"]=self.n_right_control
                    self.curr["n_left"]=len(obs["left_team"])
                    self.curr["n_right"]=len(obs["right_team"])
            if "active" in obs:
                team=None
                idx=obs["active"]
                if i<self.n_left_control:
                    team="left"
                else:
                    team="right"
                
                controls[team][idx]={}
                controls[team][idx]["sticky_actions"]=obs["sticky_actions"]
                if actions is not None:
                    controls[team][idx]["action"]=actions[i]
        self.curr["controls"]=controls
        if self.no_frame and "frame" in obs:
            self.curr.pop("frame")
        if "frame" in obs:
            self.curr["frame"]=self.compress_frame(self.curr["frame"],(960,540))

    def compress_frame(self,frame,resize):
        # TODO(jh): save as a video. see official codes in observation_processor.py
        image=Image.fromarray(frame,mode="RGB").resize(resize)
        with io.BytesIO() as output:
            image.save(output, format="jpeg",optimize=True,quality=99)
            contents = output.getvalue()
        return contents
            
    def save(self,fp):
        import os
        dir,_=os.path.split(fp)
        os.makedirs(dir,exist_ok=True)
        with open(fp,"wb") as f:
            pickle.dump(self,f)
    
    @staticmethod   
    def load(fp):
        with open(fp,"rb") as f:
            return pickle.load(f)
        
    @staticmethod
    def load_from_official_trace(dump_file,no_frame=False):
        import six.moves.cPickle
        dump = []
        with open(dump_file, 'rb') as in_fd:
            while True:
                try:
                    step = six.moves.cPickle.load(in_fd)
                except EOFError:
                    break
                dump.append(step)
                
        from gfootball.env.football_action_set import full_action_set
        
        action2idx={action:idx for idx,action in enumerate(full_action_set)}
                
        tracer=MatchTracer(no_frame)
        for step,data in enumerate(dump):
            action=data["debug"]["action"]
            action=[action2idx[a] for a in action]
            observation=data["observation"]
            d={}
            d.update(observation)
            left_agent_controlled_player=d.pop("left_agent_controlled_player")
            right_agent_controlled_player=d.pop("right_agent_controlled_player")
            left_agent_sticky_actions=d.pop("left_agent_sticky_actions")
            right_agent_sticky_actions=d.pop("right_agent_sticky_actions")
            left_team_roles=d.pop("left_team_roles")
            right_team_roles=d.pop("right_team_roles")
            if tracer.no_frame:
                d.pop("frame")
            if step==0:
               tracer.n_left_control=len(left_agent_controlled_player)
               tracer.n_right_control=len(right_agent_controlled_player)
               d["n_left_control"]=tracer.n_left_control
               d["n_right_control"]=tracer.n_right_control
               d["n_left"]=len(left_team_roles)
               d["n_right"]=len(right_team_roles)
               d["left_team_roles"]=left_team_roles
               d["right_team_roles"]=right_team_roles
            controls={"left":{},"right":{}}
            for i,(idx,sticky) in enumerate(zip(left_agent_controlled_player,left_agent_sticky_actions)):  
                controls["left"][idx]={}             
                controls["left"][idx]["sticky_actions"]=sticky
                if len(action)>0:
                    controls["left"][idx]["action"]=action[i]
            for i,(idx,sticky) in enumerate(zip(right_agent_controlled_player,right_agent_sticky_actions)):
                controls["right"][idx]={}               
                controls["right"][idx]["sticky_actions"]=sticky
                if len(action)>0:
                    controls["right"][idx]["action"]=action[i+tracer.n_left_control]
            d["controls"]=controls
            tracer.data[step]=d
        return tracer
