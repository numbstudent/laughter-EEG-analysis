
import numpy as np
import pickle
fn = ['Z:/nani/experiment/cra/funnylaugh_yes.pkl',
'Z:/nani/experiment/cra/funnylaugh_no.pkl',
'Z:/nani/experiment/ila/funnylaugh_yes.pkl',
'Z:/nani/experiment/ila/funnylaugh_no.pkl',
'Z:/nani/experiment/ovi/funnylaugh_yes.pkl',
'Z:/nani/experiment/ovi/funnylaugh_no.pkl',
'Z:/nani/experiment/aldoh/funnylaugh_yes.pkl',
'Z:/nani/experiment/aldoh/funnylaugh_no.pkl',
'Z:/nani/experiment/vyn/funnylaugh_yes.pkl',
'Z:/nani/experiment/vyn/funnylaugh_no.pkl',
'Z:/nani/experiment/gav/funnylaugh_yes.pkl',
'Z:/nani/experiment/gav/funnylaugh_no.pkl',
'Z:/nani/experiment/hrz/funnylaugh_yes.pkl',
'Z:/nani/experiment/hrz/funnylaugh_no.pkl',
'Z:/nani/experiment/rijuu/funnylaugh_yes.pkl',
'Z:/nani/experiment/rijuu/funnylaugh_no.pkl',
'Z:/nani/experiment/nature/funnylaugh_yes.pkl',
 'Z:/nani/experiment/nature/funnylaugh_no.pkl',
 'Z:/nani/experiment/manai/funnylaugh_yes.pkl',
 'Z:/nani/experiment/manai/funnylaugh_no.pkl',
 'Z:/nani/experiment/skot/funnylaugh_yes.pkl',
 'Z:/nani/experiment/skot/funnylaugh_no.pkl',
 'Z:/nani/experiment/sinlo/funnylaugh_yes.pkl',
 'Z:/nani/experiment/sinlo/funnylaugh_no.pkl',
 'Z:/nani/experiment/cips/funnylaugh_yes.pkl',
 'Z:/nani/experiment/cips/funnylaugh_no.pkl',
 'Z:/nani/experiment/fira/funnylaugh_yes.pkl',
 'Z:/nani/experiment/fira/funnylaugh_no.pkl',
 'Z:/nani/experiment/ovi2/funnylaugh_yes.pkl',
 'Z:/nani/experiment/ovi2/funnylaugh_no.pkl',
 'Z:/nani/experiment/cdef/funnylaugh_yes.pkl',
 'Z:/nani/experiment/cdef/funnylaugh_no.pkl']

ds1 = []
ds2 = []
ds1 = np.array(ds1)
ds2 = np.array(ds2)
for x in range(0,len(fn)):
     with open(fn[x], 'rb') as f:  # Python 3: open(..., 'rb')
         data = pickle.load(f)
     data = data[0]
     if x%2==0:
         if len(ds1)==0:
             ds1 = data
         else:
             print(np.array(data).shape)
             ds1 = np.concatenate((ds1,data),axis=0)
     elif x%2==1:
         if len(ds2)==0:
             ds2 = data
         else:
             print(np.array(data).shape)
             ds2 = np.concatenate((ds2,data),axis=0)

print(np.array(ds1).shape)
print(np.array(ds2).shape)
with open('Z:/nani/experiment/funny_laugh_yes.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ds1],f)
with open('Z:/nani/experiment/funny_laugh_no.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([ds2],f)
