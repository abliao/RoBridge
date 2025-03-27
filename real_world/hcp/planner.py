import base64
from openai import OpenAI
import os
import cv2
import parse
import re
import json
import parse
import numpy as np
import time
from datetime import datetime
from .prompts import *

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class Planner:
    def __init__(self,cfg):
        self.client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.config = {
            'model': cfg.model,
            'temperature': cfg.temperature,
            'max_tokens': cfg.max_tokens,
        }

    def _build_prompt(self, image_path, instruction):
        img_base64 = encode_image(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": instruction
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages
    
    def generate(self, img_path, instruction):
        messages = self._build_prompt(img_path, instruction)
        n=5
        while n>0:
            try:
                stream = self.client.chat.completions.create(model=self.config['model'],
                                                                messages=messages,
                                                                temperature=self.config['temperature'],
                                                                max_tokens=self.config['max_tokens'],
                                                                stream=True)
                output = ""
                start = time.time()
                for chunk in stream:
                    print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
                    if chunk.choices[0].delta.content is not None:
                        output += chunk.choices[0].delta.content
                print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
                return output
            except:
                n-=1
    
    def task_planning(self,img_path,task):
        instr = task_planning_prompt.format(task)
        output = self.generate(img_path,instr)
        lines = output.split('\n')
        template = "{index:d}. ({task}, {action}, {object}, {target})."
        actions = []
        for line in lines:
            line = line.lower()
            result = parse.parse(template, line)
            if result:
                index = result['index']
                task = result['task']
                action = result['action']
                obj = result['object']
                target = result['target']
                actions.append({'task':task,'action':action,'obj':obj,'target':target})

        return actions
    
    def get_IOR(self,img_path,task):
        instr = IOR_prompt.format(task)
        output = self.generate(img_path,instr)
        lines = output.split('\n')
        Action_template = 'Action: {Action}'
        Gripper_template = 'Gripper: {Gripper}'
        Object_template = 'Object: {Object}'
        Target_template = 'Target: {Target}'
        Constraint_template = "Constraint: ({x}, {y}, {z})"
        pattern = r'.*:\s*(?P<obj_name>.+?)\(marked\s+"(?P<number>\d+)"\)'
        IOR={}
        for i in range(3):
            try:
                for line in lines:
                    line = line.strip()
                    action_result = parse.parse(Action_template, line)
                    gripper_result = parse.parse(Gripper_template, line)
                    object_result = parse.parse(Object_template, line)
                    target_result = parse.parse(Target_template, line)
                    constraint_result = parse.parse(Constraint_template, line)
                    if action_result:
                        IOR['Action'] = action_result['Action'].lower()
                    elif gripper_result:
                        match = re.match(pattern, line)
                        if match is not None:
                            obj_name = match.group('obj_name').strip().lower()
                            number = match.group('number').lower()
                            IOR['Gripper']={'obj_name':obj_name,'number':number}
                        else:
                            IOR['Gripper']=None
                    elif object_result:
                        match = re.match(pattern, line)
                        obj_name = match.group('obj_name').strip().lower()
                        number = match.group('number').lower()
                        IOR['Object']={'obj_name':obj_name,'number':number}
                    elif target_result:
                        match = re.match(pattern, line)
                        if match is not None:
                            obj_name = match.group('obj_name').strip().lower()
                            number = match.group('number').lower()
                            IOR['Target']={'obj_name':obj_name,'number':number}
                        else:
                            IOR['Target']=None
                    elif constraint_result:
                        x = float(constraint_result['x'])
                        y = float(constraint_result['y'])
                        z = float(constraint_result['z'])
                        IOR['Constraint'] = np.array([x,y,z]).astype(np.float32)
            except Exception as e:
                print(f'error {e}')
            if 'Action' not in IOR or 'Object' not in IOR:
                print('output',output)
                print('IOR',IOR)
                time.sleep(1)
            else:
                break
        return IOR
    
    def check_finish(self,img_path,task,gripper_open):
        instr = check_finish_prompt.format(task,gripper_open)
        output = self.generate(img_path,instr)
        lines = output.split('\n')
        State_template = 'State: {State}'
        Reason_template = 'Reason: {Reason}'
        check_result={}
        for line in lines:
            state_result = parse.parse(State_template, line)
            reason_result = parse.parse(Reason_template, line)

            if state_result:
                check_result['State'] = state_result['State'].lower()
            elif reason_result:
                check_result['Reason'] = reason_result['Reason'].lower()
        if 'State' not in check_result:
            print('check_finish output',output)
        return check_result