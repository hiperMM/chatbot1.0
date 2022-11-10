import requests
from bs4 import BeautifulSoup

url1 = 'https://www.geeksforgeeks.org/natural-language-processing-overview/'
url2 = 'https://www.ibm.com/cloud/learn/natural-language-processing'
url3 = 'https://www.oracle.com/artificial-intelligence/what-is-natural-language-processing/'
url4 = 'https://www.deeplearning.ai/resources/natural-language-processing/'
url5 = 'https://monkeylearn.com/blog/natural-language-processing-techniques/'

page1 = requests.get(url1)
page2 = requests.get(url2)
page3 = requests.get(url3)
page4 = requests.get(url4)
page5 = requests.get(url5)

bsPage1 = BeautifulSoup(page1.text, "html.parser")
bsPage2 = BeautifulSoup(page2.text, "html.parser")
bsPage3 = BeautifulSoup(page3.text, "html.parser")
bsPage4 = BeautifulSoup(page4.text, "html.parser")
bsPage5 = BeautifulSoup(page5.text, "html.parser")

titulo1 = bsPage1.findAll('h1')
titulo1 = titulo1[0].get_text()
print("O titulo do Texto 1 é: " + titulo1)
#------------------------------------------------------------------
titulo2 = bsPage2.findAll('h1')
titulo2 = titulo2[0].get_text()
print("O titulo do Texto 2 é: " + titulo2)
#------------------------------------------------------------------
titulo3 = bsPage3.findAll('h1')
titulo3 = titulo3[0].get_text()
print("O titulo do Texto 3 é: " + titulo3)
#------------------------------------------------------------------
titulo4 = bsPage4.findAll('h1')
titulo4 = titulo4[0].get_text()
print("O titulo do Texto 4 é: " + titulo4)
#------------------------------------------------------------------
titulo5 = bsPage5.findAll('h1')
titulo5 = titulo5[0].get_text()
print("O titulo do Texto 5 é: " + titulo5)

paragrafosTexto1 = bsPage1.findAll('p')
paragrafosTexto1 = [paragrafosTexto1[i].get_text() for i in range(0, len(paragrafosTexto1))]
#----------------------------------------------------------------------------------------------------------------------
paragrafosTexto2 = bsPage2.findAll('p')
paragrafosTexto2 = [paragrafosTexto2[i].get_text() for i in range(0, len(paragrafosTexto2))]
#----------------------------------------------------------------------------------------------------------------------
paragrafosTexto3 = bsPage3.findAll('p')
paragrafosTexto3 = [paragrafosTexto3[i].get_text() for i in range(0, len(paragrafosTexto3))]
#----------------------------------------------------------------------------------------------------------------------
paragrafosTexto4 = bsPage4.findAll('p')
paragrafosTexto4 = [paragrafosTexto4[i].get_text() for i in range(0, len(paragrafosTexto4))]
#----------------------------------------------------------------------------------------------------------------------
paragrafosTexto5 = bsPage5.findAll('p')
paragrafosTexto5 = [paragrafosTexto5[i].get_text() for i in range(0, len(paragrafosTexto5))]

textoPage1 = ''.join([paragrafoTexto1 for paragrafoTexto1 in paragrafosTexto1])
#----------------------------------------------------------------------------------------------------------------------
textoPage2 = ''.join([paragrafoTexto2 for paragrafoTexto2 in paragrafosTexto2])
#----------------------------------------------------------------------------------------------------------------------
textoPage3 = ''.join([paragrafoTexto3 for paragrafoTexto3 in paragrafosTexto3])
#----------------------------------------------------------------------------------------------------------------------
textoPage4 = ''.join([paragrafoTexto4 for paragrafoTexto4 in paragrafosTexto4])
#----------------------------------------------------------------------------------------------------------------------
textoPage5 = ''.join([paragrafoTexto5 for paragrafoTexto5 in paragrafosTexto5])

import re
import nltk
nltk.download('punkt')


#----------------------------------------------------------------------------------------------------------------------
dataset1 = nltk.sent_tokenize(str(textoPage1))

for i in range (len(dataset1)):
    dataset1[i] = dataset1[i].lower()
    dataset1[i] = re.sub(r'\W+', ' ', dataset1[i])
    dataset1[i] = re.sub(r's+', ' ', dataset1[i])
#----------------------------------------------------------------------------------------------------------------------
dataset2 = nltk.sent_tokenize(str(textoPage2))

for i in range (len(dataset2)):
    dataset2[i] = dataset2[i].lower()
    dataset2[i] = re.sub(r'\W+', ' ', dataset2[i])
    dataset2[i] = re.sub(r's+', ' ', dataset2[i])
#----------------------------------------------------------------------------------------------------------------------
dataset3 = nltk.sent_tokenize(str(textoPage3))

for i in range (len(dataset3)):
    dataset3[i] = dataset3[i].lower()
    dataset3[i] = re.sub(r'\W+', ' ', dataset3[i])
    dataset3[i] = re.sub(r's+', ' ', dataset3[i])
#----------------------------------------------------------------------------------------------------------------------
dataset4 = nltk.sent_tokenize(str(textoPage4))

for i in range (len(dataset4)):
    dataset4[i] = dataset4[i].lower()
    dataset4[i] = re.sub(r'\W+', ' ', dataset4[i])
    dataset4[i] = re.sub(r's+', ' ', dataset4[i])
#----------------------------------------------------------------------------------------------------------------------
dataset5 = nltk.sent_tokenize(str(textoPage5))

for i in range (len(dataset5)):
    dataset5[i] = dataset5[i].lower()
    dataset5[i] = re.sub(r'\W+', ' ', dataset5[i])
    dataset5[i] = re.sub(r's+', ' ', dataset5[i])
#----------------------------------------------------------------------------------------------------------------------

megaTexto = dataset1 + dataset2 + dataset3 + dataset4 + dataset5

ContadorDePalavras = {}
for dado in megaTexto:
    palavras = nltk.word_tokenize(dado)
    for palavras in palavras:
        if palavras not in ContadorDePalavras.keys():
            ContadorDePalavras[palavras] = 1
        else:
            ContadorDePalavras[palavras] += 1
            
print (ContadorDePalavras)
#----------------------------------------------------------------------------------------------------------------------
import heapq

frequenciaPalavras = heapq.nlargest(200, ContadorDePalavras, key=ContadorDePalavras.get)
print (frequenciaPalavras)

#----------------------------------------------------------------------------------------------------------------------

import numpy as np

matriz = []

for dado in megaTexto:
  vetor = []
  for palavra in frequenciaPalavras:
            if palavra in nltk.word_tokenize(dado):
              vetor.append(1) # se a palavra está contida no documento coloca-se 1
            else:
              vetor.append(0) # se a palavra não está contida no documento coloca-se 0
  matriz.append(vetor)
    
bagOfWords = np.asarray(matriz)
print (bagOfWords)
