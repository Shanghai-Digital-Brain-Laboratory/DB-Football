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

import tkinter as tk
import json
import os
from functools import partial

import numpy as np
from PIL import Image,ImageTk
from tkinter.messagebox import showerror, showwarning, showinfo
from tkinter import ttk
import pandas as pd
from .translation_en import *
from ...tools import geometry

class BallInfoTable(ttk.Treeview):
    def __init__(self,parent,team,obs,step=0) -> None:
        self.team=team
        df=self.get_df(obs,self.team,step)
        columns=list(df.columns)
        super().__init__(parent,columns=columns,show="headings",selectmode ='browse')
        for i in range(len(df.columns)):
            self.column(columns[i], anchor=tk.CENTER,width=50)
            self.heading(columns[i], text=columns[i], anchor=tk.CENTER)
        # self.column("速度/0.1s",width=100)
        # self.column("坐标",width=100)
            
    def update_table(self,obs,step):
        for i in self.get_children():
            self.delete(i)
        df=self.get_df(obs,self.team,step)
        for i in range(len(df)):
            data=list(df.iloc[i])
            self.insert("",tk.END,values=data,)
    
    def get_df(self,obs,team,step):
        if step==0:
            self.left_team_roles=obs["left_team_roles"]
            self.right_team_roles=obs["right_team_roles"]
            self.n_left=obs["n_left"]
            self.n_right=obs["n_right"]
            self.n_left_control=obs["n_left_control"]
            self.n_right_control=obs["n_right_control"]
            
        coords=np.round(geometry.tpos(obs["ball"]),2)
        coord_speeds=np.round(geometry.get_coord_speed(obs["ball_direction"])/geometry.FPS,2)
        speed=np.round(geometry.get_speed(obs["ball_direction"])/geometry.FPS,2)
        rotations=np.round(obs["ball_rotation"],2)
      
        rotations=["{},{},{}".format(*rotations)]
        
        
        data=[
            ("rotations",rotations),
            ("speed/0.1s",speed),
            ("speed_x/0.1s",coord_speeds[0]),
            ("speed_y/0.1s",coord_speeds[1]),
            ("speed_z/0.1s",coord_speeds[2]),
            ("coord_x",[coords[0]]),
            ("coord_y",[coords[1]]),
            ("coord_z",[coords[2]])
        ]
        # if team=="right":
        #     data.reverse()
        df=pd.DataFrame(
            dict(data)
        )
        
        return df
    
    def pad(self,arr,width=4):
        ret=[str(d).rjust(width) for d in arr]
        return ret