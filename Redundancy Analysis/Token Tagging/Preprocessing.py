import javalang
import json

# read a file and return the array
def read_file(file_name):
    line_character_arr = []
    with open(file_name, 'r') as file:
        for line in file:
            # line.strip()
            # if not line.strip(): 
            #     continue
            line_character_arr.append(line.lstrip())
        return line_character_arr


token = read_file("code file")
code_result = []

# use javalang to convert java code to token label
for i in range(len(token)):
    tokens = list(javalang.tokenizer.tokenize(token[i]))
    # print(tokens)
    temp_result = []

    for j in range(len(tokens)):
        x = str(type(tokens[j]))
        y = x.split('.')
        y = y[-1]
        z = y.split('\'')
        # print([tokens[j].value,z[0]])
        temp_result.append(tuple([tokens[j].value,z[0]]))
    
    code_result.append(temp_result)

# dictionary to convert javalang output type to the format of training 
dictionary = {"::": "DOUBLECOLON","--":"DOUBLEMINUS","++":"DOUBLEPLUS","false":"BOOL","true":"BOOL","Modifier":"MODIFIER", "BasicType":"TYPE", "null":"IDENT","Keyword": "KEYWORD", "Identifier": "IDENT","DecimalInteger":"NUMBER","DecimalFloatingPoint":"NUMBER",
              "String":"STRING", "(": "LPAR", ")": "RPAR","[":"LSQB", "]":"RSQB",",":"COMMA", "?":"CONDITIONOP",
                  ";":"SEMI","+":"PLUS","-":"MINUS","*":"STAR","/":"SLASH", ".": "DOT",  "=": "EQUAL",":": "COLON", 
                  "|":"VBAR","&":"AMPER", "<":"LESS",">":"GREATER","%":"PERCENT","{":"LBRACE","}":"RBRACE",
                   "==":"EQEQUAL","!=":"NOTEQUAL","<=":"LESSEQUAL",">=":"GREATEREQUAL", "~":"TILDE","^":"CIRCUMFLEX",
                   "<<":"LEFTSHIFT",">>":"RIGHTSHIFT", "**":"DOUBLESTAR","+=":"PLUSEUQAL","-=":"MINEQUAL","*=":"STAREQUAL",
                   "/=":"SLASHEQUAL","%=":"PERCENTEQUAL","&=":"AMPEREQUAL","|=":"VBAREQUAL","^=":"CIRCUMFLEXEQUAL",
                   "<<=":"LEFTSHIFTEQUAL",">>=":"RIGHTSHIFTEQUAL","**=":"DOUBLESTAREQUAL","//":"DOUBLESLASH","//=":"DOUBLESLASHEQUAL",
                   "@":"AT","@=":"ATEQUAL","->":"RARROW","...":"ELLIPSIS",":=":"COLONEQUAL","&&":"AND","!":"NOT","||":"OR"}


final_result = []

# create empty array for final result
for i in code_result:
    temp_result = []
    
    for j in range(len(i)):
        temp_result.append([])
    
    final_result.append(temp_result)

# function convert takes an input of tuples (token, type) and modify to format for training
def convert(info):
    result = ""
    if info[1] == "Modifier" or info[1] == "Keyword" or info[1] == "BasicType" or info[1] == "Identifier" or info[1] == "DecimalInteger" or info[1] == "DecimalFloatingPoint" or info[1] == "String":
        result = dictionary[info[1]]
        return result
    else:
        if info[0] in dictionary:
            result = dictionary[info[0]]
            return result
        else:
            return "UNKNOWN"

# store the modified type into the final result
for i in range(len(code_result)): 
    for j in range(len(code_result[i])):
        res = ""
#         res = convert(code_result[i][j])
        res = code_result[i][j][1]
        final_result[i][j] = res

# write the result into a final_result.txt
f = open("valid.in", "w")
for arr in code_result:
    if arr != []:
        for i in arr:
            f.write(i[0] + " ")
        f.write("\n")
f.close()

# write the result into a final_result.txt
f = open("valid.label", "w")
for arr in final_result:
    if arr != []:
        for i in arr:
            f.write(i + " ")
        f.write("\n")
f.close()