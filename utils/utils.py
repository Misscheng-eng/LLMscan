'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-20 13:10:39
LastEditors: zhangmengling zhangmengdi1997@126.com
LastEditTime: 2024-07-23 17:12:05
FilePath: /mengdizhang/LLM-LieDetector/utils/utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#from fastchat.model import get_conversation_template


def load_conversation_template(template_name):
    """加载对话模板"""
    try:
        from fastchat.conversation import get_conversation_template
        conv_template = get_conversation_template(template_name)
        return conv_template
    except ImportError:
        print("⚠️  fastchat 未安装，使用默认模板")
        return create_default_template(template_name)
    except Exception as e:
        print(f"⚠️  加载模板失败: {e}，使用默认模板")
        return create_default_template(template_name)

def create_default_template(template_name):
    """创建默认对话模板"""
    class DefaultConversation:
        def __init__(self, name):
            self.name = name
            self.system_message = "You are a helpful AI assistant."
            self.roles = ["User", "Assistant"]
            self.messages = []
            self.sep = "\n"
            self.sep2 = "\n"
        
        def append_message(self, role, message):
            self.messages.append([role, message])
        
        def get_prompt(self):
            ret = self.system_message + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
            return ret
        
        def copy(self):
            import copy
            return copy.deepcopy(self)
    
    return DefaultConversation(template_name)


def generate_input(conv_template,prompt,adv_suffix=None):   
    conv_template.messages = []
    if adv_suffix is not None:
        conv_template.append_message(conv_template.roles[0], f"{prompt} {adv_suffix}")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt()
    else:
        conv_template.append_message(conv_template.roles[0], f"{prompt}.")
        conv_template.append_message(conv_template.roles[1], None)
        result = conv_template.get_prompt() + " "
    return result

def prepare_prompt(question, instruction=None):
    if instruction == None:
        prompt = "Question: " + question + "\n" + "Answer:"
    else:
        prompt = instruction + "\n" + "Question: " + question + "\n" + "Answer: "
    return prompt

def prepare_prompt_choice(question, choices, instruction=None):
    if instruction == None:
        question_prompt = "Question: " + question + "\n"
        choices_prompt = ''.join([str(i) + ") " + str(choices[i]) + "\n" for i in range(0, len(choices))])
        prompt = question_prompt + choices_prompt + "\n" + "Answer:"
    else:
        question_prompt = instruction + "\n" + "Question: " + question + "\n"
        choices_prompt = ''.join([str(i) +  ") " + str(choices[i]) + "\n" for i in range(0, len(choices))])
        prompt = question_prompt + choices_prompt + "\n" + "Answer:"
    return prompt

def prepare_prompt_completion(temp, values, instruction=None):
    """
    temp:, eg., "Say something {} when {}"
    values: e.g., [prompt_type, question]
    """
    prompt = temp.format(*values) + "\n" + "Example: "
    return prompt