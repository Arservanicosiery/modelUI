from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout,QHBoxLayout, QLineEdit,QTextEdit
from PyQt5.QtGui import QColor, QPixmap, QIcon, QPalette
from PyQt5.QtGui import (QPen, QPainter, QBrush, 
                         QLinearGradient, QConicalGradient, QRadialGradient)
from PyQt5.QtCore import QObject, pyqtSignal,QSize
from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import QMainWindow,QTextBrowser
from PyQt5.QtGui import QFont, QFontMetrics
from PyQt5.QtCore import Qt
from PyQt5 import  QtWidgets
import ollama
import sys
import cv2
import dlib
import numpy as np
import math
import wave
import time
import pyaudio
import pyqtgraph as pg  
from pyqtgraph.Qt import QtCore, QtGui  
import psutil  
import numpy as np  
from imutils import face_utils
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QLabel,QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer,QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QMessageBox
import listen
from monitor import monitor
import socket
from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.embeddings import OllamaEmbeddings
from moviepy.editor import VideoFileClip, AudioFileClip
# 全局变量，可调整进度条的大小
PROGRESS_WIDTH = 100
PROGRESS_HEIGHT = 100

video = VideoFileClip("media.mp4")  
audio = AudioFileClip("media.mp3")  
# 全局变量，可调整进度条动画速度
PROGRESS_SPEED = 50  # 毫秒
from untitled import Ui_MainWindow     #这两个是聊天界面的
from mouth_check import App    #视觉识别
from circle_button import RoundProgress   #圆形进度条的

def call_llama_model(myprompt):
    # 这里是函数实现
    print(myprompt)
    response = ollama.generate(
        model='llama2-chinese',
        prompt=myprompt,
        stream=False
    )

    return response['response']

class StyledWidget(QWidget):
    def __init__(self, title, gradient_start, gradient_end, radius=10):
        super().__init__()
        self.title = title
        self.gradient_start = gradient_start
        self.gradient_end = gradient_end
        self.radius = radius
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)

        # 设置背景渐变色
        palette = QPalette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, self.gradient_start)
        gradient.setColorAt(1.0, self.gradient_end)
        palette.setBrush(QPalette.Background, gradient)
        self.setAutoFillBackground(True)
        self.setPalette(palette)

        # 设置圆角
        self.setStyleSheet(f"border-radius: {self.radius}px;")

class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Main App')

        # 创建三个按钮
        self.chat_button = QPushButton('聊天模式', self)
        self.video_button = QPushButton('虚拟形象', self)
        self.learn_button = QPushButton('知识库嵌入', self)
        self.model_button = QPushButton('更新模型',self)
        self.web_button=QPushButton('网页总结',self)
        self.serve_button = QPushButton('服务器',self)

        # 将按钮放置在垂直布局中
        self.old_layout = QVBoxLayout()
        self.new_layout=QHBoxLayout()
        self.old_layout.addWidget(self.video_button)
        self.old_layout.addWidget(self.learn_button)
        self.old_layout.addWidget(self.model_button)
        self.old_layout.addWidget(self.web_button)
        self.old_layout.addWidget(self.serve_button)
        self.new_layout.addWidget(self.chat_button)
        self.new_layout.addLayout(self.old_layout)
        
        self.setLayout(self.new_layout)


    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Chat Mode')

        # 创建三个按钮
        self.local_model_button = QPushButton('本地模型', self)
        self.text_button = QPushButton('文心一言', self)
        self.xunfei_button = QPushButton('讯飞星火', self)
        self.chat_back_button=QPushButton('back',self)

        # 将按钮放置在垂直布局中
        layout = QVBoxLayout()
        layout.addWidget(self.local_model_button)
        layout.addWidget(self.text_button)
        layout.addWidget(self.xunfei_button)
        layout.addWidget(self.chat_back_button)
        self.setLayout(layout)

class VideoMode(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Video Mode')

        # 创建两个按钮
        self.open_file_button = QPushButton('数字虚拟人', self)
        self.video_back_button=QPushButton('back',self)

        # 将按钮放置在垂直布局中
        layout = QVBoxLayout()
        layout.addWidget(self.open_file_button)
        layout.addWidget(self.video_back_button)
        self.setLayout(layout)

class LearnMode(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Learn Mode')

        # 创建一个文本框
        self.text_edit = QLineEdit(self)
        self.question_edit = QLineEdit(self)
        self.text_edit_button=QPushButton('知识库嵌入',self)
        self.learn_send_button = QPushButton('发送',self)
        self.learn_back_button=QPushButton('back',self)

        # 将文本框放置在垂直布局中
        self.layout = QVBoxLayout()
        self.learn_layout=QHBoxLayout()
        self.learn_layout.addWidget(self.question_edit)
        self.learn_layout.addWidget(self.learn_send_button)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.text_edit_button)
        self.layout.addLayout(self.learn_layout)
        self.layout.addWidget(self.learn_back_button)

        self.setLayout(self.layout)
        self.text_edit_button.clicked.connect(self.text_edit_button_clicked)  
        self.learn_send_button.clicked.connect(self.learn_send_button_clicked)  

    def text_edit_button_clicked(self):
        text=self.text_edit.text()
        self.insert(text)

    def learn_send_button_clicked(self):
        question=self.question_edit.text()
        response=self.generate(question)
        self.text_edit.setText(response)

    def insert(self, text):
        model_local = ChatOllama(model="llama2-chinese")

        # 1. 读取文件并分词
        '''documents = TextLoader("./三国演义.txt").load()'''
        self.loader = TextLoader("./三国演义.txt",encoding="utf-8")
        self.documents = self.loader.load()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
        doc_splits = text_splitter.split_documents(self.documents)

        # 2. 嵌入并存储
        embeddings = OllamaEmbeddings(model='nomic-embed-text')
        vectorstore = DocArrayInMemorySearch.from_documents(doc_splits, embeddings)
        retriever = vectorstore.as_retriever()
        print("知识库嵌入")

    def generate(self, question):
        model_local = ChatOllama(model="llama2-chinese")
        template = """Answer the question based only on the following context:
        {context}
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": 'chat', "question": RunnablePassthrough()}
            | prompt
            | model_local
            | StrOutputParser()
        )
        answer=chain.invoke(question)
        return answer

class Set_question:
    def __init__(self):
        self.message_id = 0  # 用于跟踪消息的 ID，以便区分左右消息

    def set_return(self, right_ico, left_ico, text, dir):
        self.widget = QtWidgets.QWidget(self.scrollAreaWidgetContents)
        self.widget.setLayoutDirection(dir)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        # 根据方向选择头像
        if dir == QtCore.Qt.RightToLeft:
            ico = QtGui.QPixmap("right_avatar.jpg")
            alignment = QtCore.Qt.AlignRight
        else:
            ico = QtGui.QPixmap("left_avatar.jpg")
            alignment = QtCore.Qt.AlignLeft
        
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setMaximumSize(QtCore.QSize(50, 50))
        self.label.setText("")
        self.label.setPixmap(ico)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)

        self.textBrowser = QtWidgets.QTextBrowser(self.widget)
        self.textBrowser.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.textBrowser.setStyleSheet("padding:10px;\n"
                                        "background-color: rgba(71,121,214,20);\n"
                                        "font: 16pt \"黑体\";")
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setText(text)
        self.textBrowser.setMinimumSize(QtCore.QSize(0, 0))
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.textBrowser.setAlignment(alignment)  # 设置文本对齐方式

        self.horizontalLayout.addWidget(self.textBrowser)
        self.verticalLayout.addWidget(self.widget)

        #model() 函数生成的新消息
        new_message = self.call_llama_model(text)  
        self.add_new_message(new_message, QtCore.Qt.LeftToRight if dir == QtCore.Qt.RightToLeft else QtCore.Qt.RightToLeft)


    def call_llama_model(self, prompt):
        # 调用 Llama 模型生成响应文本
        response = call_llama_model(prompt)  
        return response

    def add_new_message(self, new_message, dir):
        # 添加新的消息到聊天窗口
        right_ico = QtGui.QPixmap("right_avatar.jpg")  # 右侧头像
        left_ico = QtGui.QPixmap("left_avatar.jpg")    # 左侧头像
        self.set_return(right_ico, left_ico, new_message, dir)  # 调用 set_return() 方法显示新消息

class MainWindow(QMainWindow,Ui_MainWindow):

    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.sum=0                                                  #气泡数量
        self.widgetlist = []                                        #记录气泡
        self.text = ""                                              # 存储信息                         # 头像
        #设置聊天窗口样式 隐藏滚动条
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        # 信号与槽
        self.pushButton.clicked.connect(self.create_widget)         #创建气泡
        self.pushButton.clicked.connect(self.set_widget)            #修改气泡长宽
        self.plainTextEdit.undoAvailable.connect(self.Event)        #监听输入框状态
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.rangeChanged.connect(self.adjustScrollToMaxValue)
        self.window_back_button = QPushButton('back', self) #监听窗口滚动条范围


    # 回车绑定发送
    def Event(self):
        if not self.plainTextEdit.isEnabled():  #这里通过文本框的是否可输入
            self.plainTextEdit.setEnabled(True)
            self.pushButton.click()
            self.plainTextEdit.setFocus()

    #创建气泡
    def create_widget(self):
        self.text=self.plainTextEdit.toPlainText()
        self.plainTextEdit.setPlainText("")
        self.sum += 1
        if self.sum % 2:   # 根据判断创建左右气泡
            Set_question.set_return(self, self.icon, self.text,QtCore.Qt.LeftToRight)    # 调用new_widget.py中方法生成左气泡
            QApplication.processEvents()                                # 等待并处理主循环事件队列
        else:
            Set_question.set_return(self, self.icon, self.text,QtCore.Qt.RightToLeft)   # 调用new_widget.py中方法生成右气泡
            QApplication.processEvents()                                # 等待并处理主循环事件队列


        # 你可以通过这个下面代码中的数组单独控制每一条气泡
        # self.widgetlist.append(self.widget)
        # print(self.widgetlist)
        # for i in range(self.sum):
        #     f=self.widgetlist[i].findChild(QTextBrowser)    #气泡内QTextBrowser对象
        #     print("第{0}条气泡".format(i),f.toPlainText())

    # 修改气泡长宽
    def set_widget(self):
        font = QFont()
        font.setPointSize(16)
        fm = QFontMetrics(font)
        text_width = fm.width(self.text)+115    #根据字体大小生成适合的气泡宽度
        if self.sum != 0:
            if text_width>632:                  #宽度上限
                text_width=int(self.textBrowser.document().size().width())+100  #固定宽度
            self.widget.setMinimumSize(text_width,int(self.textBrowser.document().size().height())+ 40) #规定气泡大小
            self.widget.setMaximumSize(text_width,int(self.textBrowser.document().size().height())+ 40) #规定气泡大小
            self.scrollArea.verticalScrollBar().setValue(10)


    # 窗口滚动到最底部
    def adjustScrollToMaxValue(self):
        scrollbar = self.scrollArea.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

class download(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle('download new model')
        #下载新模型
        self.lb1 = QLabel('请注意，本机内存有限，下载新模型后将自动卸载原有模型',self)
        self.llama2_button=QPushButton('llama2-chinese', self)
        self.llama3_button=QPushButton('llama3',self)
        self.mistral_button=QPushButton('mistral',self)
        self.qwen_button=QPushButton('qwen',self)
        self.llama2_uncensored_button=QPushButton('llama2-uncensored',self)
        self.dl_back_button=QPushButton('back',self)

        layout = QVBoxLayout()
        layout.addWidget(self.lb1)
        layout.addWidget(self.llama2_button)
        layout.addWidget(self.llama3_button)
        layout.addWidget(self.mistral_button)
        layout.addWidget(self.qwen_button)
        layout.addWidget(self.llama2_uncensored_button)
        layout.addWidget(self.dl_back_button)
        
        self.setLayout(layout)

class tongji(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  
  
    def initUI(self):  
        # 设置窗口标题和大小  
        self.setWindowTitle('tongji')  
        self.setGeometry(100, 100, 300, 200)  
  
        # 创建一个按钮  
        self.button = QPushButton(self)  
        self.button.setIcon(QIcon('tongji.png'))
        self.button.setIconSize(QSize(200, 200))
        # 设置按钮的位置和大小  
        self.start_button=QPushButton("Let's go",self)
        # 显示窗口  
        self.show()  

class web(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  
  
    def initUI(self):  
        # 创建一个 QLineEdit 控件  
        self.line_edit = QLineEdit(self)   
  
        # 创建一个 QTextEdit 用于显示文本  
        self.send_button = QPushButton('生成',self)  
        self.text_edit = QTextEdit(self)
        self.zong_label = QLabel('总结',self)
        self.text_edit.setReadOnly(True)
        self.web_back_button = QPushButton('back',self)
        # 创建一个垂直布局  
               # 创建一个垂直布局  
        self.layout = QVBoxLayout()
        self.heng_layout=QHBoxLayout()
        self.heng_layout.addWidget(self.zong_label)
        self.heng_layout.addWidget(self.text_edit)
        self.layout.addWidget(self.line_edit)  
        self.layout.addWidget(self.send_button) 
        self.layout.addLayout(self.heng_layout)
        self.layout.addWidget(self.web_back_button)
  
        # 设置窗口的布局  
        self.setLayout(self.layout)  
  
        # 设置窗口的标题和大小  
        self.setWindowTitle('web')  
        self.setGeometry(300, 300, 250, 150)
        self.send_button.clicked.connect(self.send_clicked)  
    def summary(self, text):
        self.loader = WebBaseLoader(text)
        self.docs = self.loader.load()

        llm = Ollama(model="llama2-chinese")
        chain = load_summarize_chain(llm, chain_type="stuff")

        result = chain.run(self.docs)
        return result
    
    def send_clicked(self):  
        # 获取line_edit中的文本  
        text = self.line_edit.text()  
        # 调用summary函数并获取摘要  
        summary_text = self.summary(text)  
        # 将摘要显示在text_edit中  
        self.text_edit.setText(summary_text) 

class serve(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()  
        self.setup_monitor()  
    def initUI(self):
        main_layout = QVBoxLayout(self)
        self.labels = [  
            QLabel('CPU使用率', self),  
            QLabel('内存占用', self),  
            QLabel('剩余内存', self),  
            QLabel('内存占用百分比', self),   
        ] 
        self.lianjie_edit=QTextEdit(self)
        self.lianjie_edit.setText('0')
        self.serve_labelx=QLabel('连接数',self)
        self.serve_start_button=QPushButton('启动服务器',self)
        self.serve_label6=QLabel('连接终止后会自动保存聊天记录',self)
        self.text_edits = [  
            QTextEdit(self),  
            QTextEdit(self),  
            QTextEdit(self),  
            QTextEdit(self) 
        ]
        for label, text_edit in zip(self.labels, self.text_edits):  
            # 创建垂直布局  
            self.h_layout = QVBoxLayout()
            self.add_layout=QHBoxLayout()  
            # 添加标签和文本框到垂直布局
            self.h_layout.addWidget(label)  
            self.h_layout.addWidget(text_edit)  
            # 添加水平布局到主垂直布局  
            self.add_layout.addLayout(self.h_layout)
        self.serve_back_button = QPushButton('back',self)
        main_layout.addWidget(self.serve_label6)
        main_layout.addWidget(self.serve_labelx)
        main_layout.addWidget(self.serve_start_button)
        main_layout.addLayout(self.add_layout)
        main_layout.addWidget(self.serve_back_button)
        self.serve_start_button.clicked.connect(self.startserver())
    
    def startserver(self):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        host = '100.80.72.99'
        port = 9999

        serversocket.bind((host, port))
        serversocket.listen(5)

        print("等待客户端连接...")

        try:
            clientsocket, addr = serversocket.accept()
            print("连接地址:", addr)
            self.lianjie_edit.setText('1')

            while True:
                try:
                    message = clientsocket.recv(1024).decode('utf-8')
                    if message:
                        print("客户端消息:"+ message)
                        response_message=ollama.chat(model='llama2-chinese', messages=[{'role': 'user', 'content': message}])
                        print('输出消息：'+ response_message['message']['content'])
                        new_data=response_message['message']['content']
                        clientsocket.send((new_data+'\n').encode('utf-8'))
                        '''clientsocket.flush()'''
                except socket.error as e:
                    print("Socket error:", e)
                    break
        except KeyboardInterrupt:
            print("服务器关闭")
        finally:
            clientsocket.close()
            serversocket.close()

    def setup_monitor(self):  
        # 设置定时器，每隔0.5秒触发一次update_data方法  
        self.timer = QTimer(self)  
        self.timer.timeout.connect(self.update_data)  
        self.timer.start(500)  # 0.5秒 = 500毫秒  
  
    def monitor(self):
        mem = psutil.virtual_memory()
        # 系统总计内存

        cpu = psutil.cpu_percent(interval=0.2)

        percent = mem.used / mem.total

        # 内存总量
        total = float(mem.total) / 1024 / 1024 / 1024
        # 已经使用
        used = float(mem.used) / 1024 / 1024 / 1024
        # 剩余量
        free = float(mem.free) / 1024 / 1024 / 1024

        # cpu占用
        cpu = str(cpu)[:4]
        # 内存占用百分比
        percent = str(percent)[:6]
        used = str(used)[:4]
        free = str(free)[:4]
        total = str(total)[:4]

        # li = [cpu, total,used,free,percent]
        # print(li)

        data = {"cpu":cpu,"total":total,"used":used,"free":free,"percent":percent}
        return data
    
    def update_data(self):  
        data = self.monitor()  
        self.text_edits[0].setText(data["cpu"])  
        self.text_edits[1].setText(data["used_mem"])  
        self.text_edits[2].setText(data["free_mem"])  
        self.text_edits[3].setText(data["mem_percent"])  
        

class Controller(QObject):
    def __init__(self):
        super().__init__()
        self.model=''
        self.models=''
        self.app = QApplication([])
        self.main_app = MainApp()
        self.video_mode = VideoMode()
        self.learn_mode = LearnMode()
        self.main_mode = MainWindow()
        self.dl=download()
        self.tju=tongji()
        self.websummary=web()
        self.PCserve=serve()
        self.ddd=RoundProgress()
        self.digit=App()
        
        self.initConnections()
        self.showtj()

    def initConnections(self):
        self.tju.start_button.clicked.connect(lambda:self.showMainApp(''))
        self.main_app.chat_button.clicked.connect(self.showMainWindow)
        self.main_app.video_button.clicked.connect(self.showVideoMode)
        self.main_app.learn_button.clicked.connect(self.showLearnMode)
        self.main_app.model_button.clicked.connect(self.showdownload)
        self.main_app.web_button.clicked.connect(self.displayWeb)
        self.main_app.serve_button.clicked.connect(self.server)
        #主界面的六个按钮，每个对应一个界面，信号槽
        #前两个有点复杂一会再写

        #返回信号槽设置
        self.video_mode.video_back_button.clicked.connect(lambda:self.showMainApp('video'))
        self.learn_mode.learn_back_button.clicked.connect(lambda:self.showMainApp('learn'))
        self.websummary.web_back_button.clicked.connect(lambda:self.showMainApp('web'))
        self.PCserve.serve_back_button.clicked.connect(lambda:self.showMainApp('server'))
        self.dl.dl_back_button.clicked.connect(lambda:self.showMainApp('down'))

        self.video_mode.open_file_button.clicked.connect(self.showApp)
        '''self.digit.digit_back_button.clicked.connect(lambda:self.MainApp('digit'))'''
        
        #模型下载信号槽
        self.dl.llama2_button.clicked=(lambda:self.downmodel('llama2-chinese:latest'))
        self.dl.llama3_button.clicked=(lambda:self.downmodel('llama3'))
        self.dl.mistral_button.clicked=(lambda:self.downmodel('mistral'))
        self.dl.qwen_button.clicked=(lambda:self.downmodel('qwen:7b'))
        self.dl.llama2_uncensored_button.clicked=(lambda:self.downmodel('llama2_great'))
        
        
        


    def downmodel(self,modelkind):
        data=ollama.list()
        for model2 in data['models']:
            print (model2['name'])
            if modelkind==model2['name']:
                QMessageBox.information(self, "提示", "此模型已安装")

            elif (model2['name']!='nomic-embed-text:latest'):
                ollama.delete(model2['name'])
                self.ddd.startProgress()
                ollama.pull(modelkind)
                self.ddd.finishProgress()

    def showtj(self):
        self.tju.show()

    def displayWeb(self):
        self.main_app.hide()
        self.websummary.show()

    def server(self):
        self.main_app.hide()
        self.PCserve.show()

    def showApp(self):
        self.video_mode.hide()
        self.digit.show()

    def showMainApp(self,model):
        if model=='video':
            self.video_mode.hide()
        if model=='learn':
            self.learn_mode.hide()
        if model=='digit':
            self.digit.hide()
        if model=='down':
            self.dl.hide()
        if model=='web':
            self.websummary.hide()
        if model=='serve':
            self.PCserve.hide()
        self.main_app.show()

    def showVideoMode(self):
        self.main_app.hide()
        self.video_mode.show()

    def showLearnMode(self):
        self.main_app.hide()
        self.learn_mode.show()
    def showMainWindow(self,models):
        self.main_app.hide()
        self.main_mode.show()
    def showdownload(self):
        self.learn_mode.hide()
        self.dl.show()


    def run(self):
        self.app.exec_()

if __name__ == '__main__':
    controller = Controller()
    controller.run()
