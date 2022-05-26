#Name: Joshua Naeem 
#Roll: 22-10367
#Course: COMP 421

#import the required libraries
import rsa
import pandas

#The main function
def encryption_function(s):
    publicKey, privateKey = rsa.newkeys(512)
    ENCRYPTEDSTRING = rsa.encrypt(s.encode(),publicKey)

    #Displaying the Entered String and the encrypted string
    print("original string: ", s)
    print("encrypted string: ", ENCRYPTEDSTRING)
    
    encrypted_message_to_decrypt=ENCRYPTEDSTRING  
    DECRYPTEDSTRING = rsa.decrypt(encrypted_message_to_decrypt, privateKey).decode()
    print("decrypted string: ", DECRYPTEDSTRING)


#Calling the main function
if __name__ == '__main__':
    encryption_function(input("Enter a Message to encrypt: "))