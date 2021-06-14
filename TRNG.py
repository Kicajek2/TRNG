import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
import math
import numpy as np
from scipy.stats import entropy
from math import log, e
import pandas as pd

# Je Sen Teh , WeiJian Teng , Azman Samsudin “A True Random Number Generator
# Based on Hyperchaos and Digital Sound”

chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 22
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames


# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    data_int = np.array(struct.unpack(str(2 * chunk) + 'B', data), dtype = 'b')[::2] + 128
    frames.extend(data_int)

# Stop and close the stream 
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

bits3 = []
mask = 0b00000111
counter = 0

for x in frames:
    counter = counter + 1
    if(counter > 10000):
        value = mask & x
        bits3.append(value)

def entropy1(labels, base=None):
  value, counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

def floatToRawLongBits(value):
	return struct.unpack('Q', struct.pack('d', value))[0]

def xor(x,y):
    z = int(x, 2) ^ int(y, 2)
    return '{0:b}'.format(z).zfill(64)

def swap(z):
    tmp=""
    tmp+=(z[32:63]+z[0:31])
    return tmp

def tentmap(x):
  a = 1.99999
  if x < 0.5:
      return np.float64(a*x)
  else:
      return np.float64(a*(1-x))

def postprocessing(r):
  L = 8
  N = 256
  O = ""
  E = 0.05

  y = math.floor(L/2)
  n = N / 32

  rows, cols = (50, 50)
  x = [[0]*cols]*rows

  x[0][0] = np.float64(0.141592)
  x[0][1] = np.float64(0.653589)
  x[0][2] = np.float64(0.793238)
  x[0][3] = np.float64(0.462643)
  x[0][4] = np.float64(0.383279)
  x[0][5] = np.float64(0.502884)
  x[0][6] = np.float64(0.197169)
  x[0][7] = np.float64(0.399375)

  z = [0 for i in range(8)]
  c = 0

  while len(O) < N:
    for i in range(0, int(L - 1)):
      t = 0
      x[t][i] = ((0.071428571*r[c]) + x[t][i])*0.666666667
      #print(x[t][i])
      c = c + 1
    
    for t in (0, y - 1):
      for i in (0, L - 1):
        x[t + 1][i] = (1 - E) * tentmap(x[t][i])  +  (E/2)*(tentmap(x[t][(i+1) % L])) + tentmap(x[t][i - 1])
    
    for i in range(L):
      z[i] = floatToRawLongBits(x[y-1][i])
      z[i] = "{0:b}".format(z[i])
      x[0][i] = x[y-1][i]
    
    z[0] = xor(z[0],swap(z[4]))
    z[1] = xor(z[1],swap(z[5]))
    z[2] = xor(z[2],swap(z[6]))
    z[3] = xor(z[3],swap(z[7]))
    O+=z[0]
    O+=z[1]
    O+=z[2]
    O+=z[3]
  return O


#################################LICZENIE##################################
##VARIABLES#####

howMany = 100000
listOfNumbers = []

partOfbits3 = [0 for i in range(8)]
counter = 0
while(counter <= 8*howMany):
  for j in range(0, 8):
    partOfbits3[j] = bits3[j+counter]
  counter = counter + 8
  var = postprocessing(partOfbits3)
  for i in range(0, 32):
    listOfNumbers.append(int(var[i*8:i*8+8], 2))



#n = plt.hist(frames, bins=256, facecolor='blue', alpha=0.5, density=True)
#plt.show()

#m = plt.hist(listOfNumbers, bins=256, facecolor='blue', alpha=0.5, density=True)
#plt.show()

#print("Entropia 8bitowych liczb: %f"%entropy1(frames, 2))
#print("Entropia 256bitowych na 8 bit liczb: %f"%entropy1(listOfNumbers, 2))

#print(listOfNumbers[1])

########################################## Count-the-Ones Tests ############################ 

howManytest = 256000
listofchars = [0 for i in range(howManytest)]
listofQ5 = [0 for i in range(3125)]
listofQ4 = [0 for i in range(625)]
howmanyQ5 = [0 for i in range(3125)]
howmanyQ4 = [0 for i in range(625)]
howmanyone = 0
doQ5 = []
doQ4 = []
charlist = ["A","B","C","D","E"]
charhelp = "A"

for i in range (0, howManytest):
  if( (listOfNumbers[i] & 0b00000001) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b00000010) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b00000100) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b00001000) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b00010000) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b00100000) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b01000000) > 0 ):
    howmanyone = howmanyone + 1
  if( (listOfNumbers[i] & 0b10000000) > 0 ):
    howmanyone = howmanyone + 1

  if(howmanyone < 3):
    listofchars[i] = "A"

  if(howmanyone == 3):
    listofchars[i] = "B"

  if(howmanyone == 4):
    listofchars[i] = "C"

  if(howmanyone == 5):
    listofchars[i] = "D"

  if(howmanyone > 5):
    listofchars[i] = "E"

  howmanyone = 0

howmanyone = 0

for i in range (0, 5):
  for j in range (0, 5):
    for k in range (0, 5):
      for l in range (0, 5):
        for m in range (0, 5):
          charhelp = charlist[i]
          charhelp = charhelp + charlist[j]
          charhelp = charhelp + charlist[k]
          charhelp = charhelp + charlist[l]
          charhelp = charhelp + charlist[m]
          listofQ5[howmanyone] = charhelp
          howmanyone = howmanyone + 1

howmanyone = 0

for i in range (0, 5):
  for j in range (0, 5):
    for k in range (0, 5):
      for l in range (0, 5):
        charhelp = charlist[i]
        charhelp = charhelp + charlist[j]
        charhelp = charhelp + charlist[k]
        charhelp = charhelp + charlist[l]
        listofQ4[howmanyone] = charhelp
        howmanyone = howmanyone + 1

for i in range (0 , 255995):
  charhelp = listofchars[i] + listofchars[i+1] + listofchars[i+2] + listofchars[i+3] + listofchars[i+4]
  for j in range (0 , 3125):
    if(charhelp == listofQ5[j]):
      howmanyQ5[j]= howmanyQ5[j] + 1
      break

for i in range (0 , 255996):
  charhelp = listofchars[i] + listofchars[i+1] + listofchars[i+2] + listofchars[i+3]
  for j in range (0 , 625):
    if(charhelp == listofQ4[j]):
      howmanyQ4[j]= howmanyQ4[j] + 1
      break

for i in range (0 , 255995):
  charhelp = listofchars[i] + listofchars[i+1] + listofchars[i+2] + listofchars[i+3] + listofchars[i+4]
  for j in range (0 , 3125):
    if(charhelp == listofQ5[j]):
      howmanyQ5[j]= howmanyQ5[j] + 1
      doQ5.append(charhelp)
      break

for i in range (0 , 255996):
  charhelp = listofchars[i] + listofchars[i+1] + listofchars[i+2] + listofchars[i+3]
  for j in range (0 , 625):
    if(charhelp == listofQ4[j]):
      howmanyQ4[j]= howmanyQ4[j] + 1
      doQ4.append(charhelp)
      break

plt.title('Empiryczny rozklad Q4')
plt.xlabel('Wartosc (x)')
plt.ylabel('Czestotliwosc wystepowania(p)')
plt.hist(doQ4,bins=625,density=True)
plt.show()

plt.title('Empiryczny rozklad Q5')
plt.xlabel('Wartosc (x)')
plt.ylabel('Czestotliwosc wystepowania(p)')
plt.hist(doQ5,bins=3125,density=True)
plt.show()

