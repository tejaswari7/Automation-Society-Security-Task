from flask import Flask,request,render_template, Response
import camera
from camera import VideoCamera
import Bot
import pyrebase



app = Flask(__name__)
global temp,response


@app.route('/')
def index():
    return render_template('index.html')

config = {
        "Enter firebase details here"
              };
           # Initialize Firebase
firebase=pyrebase.initialize_app(config)
db = firebase.database()

@app.route('/temp/', methods=['GET','POST'])
def tdata():
	tmp = request.args.get('temp')
	temp = tmp
	camera.t = tmp
	return tmp

def gen(camera):
    print("\n\n4:face recog started \n\n")
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    print("\n\n3:cam on ready\n\n ")
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/audio')
def audio():
    print("\n\nSpeech\n\n ")
    return render_template('speech.html')

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)

@app.route("/get")
def chat():
    global ans
    flag = True
    while (flag == True):
        user_response = request.args.get('msg')
        user_response = user_response.lower()
        if user_response not in GREETING_INPUTS and user_response.isalpha() == False:
            camera.flat = user_response
            
        if user_response.isalpha() and user_response not in GREETING_INPUTS:    
            camera.uname = user_response
        
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                flag = False
                response = "You are welcome.."
                return response
            else:
                if (Bot.greeting(user_response) != None):
                    response = Bot.greeting(user_response)
                    ans = 1 
                    return response
                elif ans == 1:
                    response = "What is your name? "
                    ans = 0
                    return response
                else:

                    response = Bot.response(user_response)
                    Bot.sent_tokens.remove(user_response)
                    return response
        else:
            flag = False
            response = "Bye! take care"
            return response
if __name__ == '__main__':
	app.run(debug=True)
    
