VERSION 5.00
Begin VB.Form Form1 
   Caption         =   "Form1"
   ClientHeight    =   6630
   ClientLeft      =   60
   ClientTop       =   345
   ClientWidth     =   11205
   LinkTopic       =   "Form1"
   ScaleHeight     =   6630
   ScaleWidth      =   11205
   StartUpPosition =   3  'Windows 기본값
   Begin VB.CommandButton Command1 
      Caption         =   "Command1"
      Height          =   765
      Left            =   7245
      TabIndex        =   1
      Top             =   4245
      Width           =   2445
   End
   Begin VB.TextBox Text1 
      BeginProperty Font 
         Name            =   "굴림"
         Size            =   12
         Charset         =   129
         Weight          =   700
         Underline       =   0   'False
         Italic          =   0   'False
         Strikethrough   =   0   'False
      EndProperty
      Height          =   2310
      Left            =   330
      TabIndex        =   0
      Top             =   915
      Width           =   8865
   End
End
Attribute VB_Name = "Form1"
Attribute VB_GlobalNameSpace = False
Attribute VB_Creatable = False
Attribute VB_PredeclaredId = True
Attribute VB_Exposed = False
Dim i As Integer
Dim ii As Integer
Dim j As Integer
Dim k As Integer


Dim NJ As Integer
Dim NM As Integer

Dim BM As Integer
Dim q As Integer

Dim t(100, 100) As Integer
Dim W(10) As Double
Dim CT As Integer

Dim strX As String
Dim strY As String


Dim maxJ As Integer
Dim maxM(10) As Integer


Dim RES As Integer

Dim P(100, 100) As Integer
Dim RE(100, 100) As Integer

Private Sub Command1_Click()

Text1.Text = "실행중"

' 직전선행작업 입력

NJ = 5
NM = 3

For i = 1 To N
    
    For j = 1 To N
    
        P(i, j) = 0
        
    Next
Next


P(1, 3) = 1
P(1, 4) = 1
P(2, 4) = 1
P(3, 5) = 1
P(3, 6) = 1
P(4, 7) = 1
P(7, 8) = 1
P(5, 9) = 1
P(6, 9) = 1
P(6, 10) = 1
P(8, 11) = 1
P(9, 12) = 1
P(10, 13) = 1
P(11, 13) = 1
P(11, 14) = 1
P(12, 15) = 1
P(14, 16) = 1
P(15, 17) = 1
P(15, 18) = 1
P(16, 19) = 1
P(17, 20) = 1
P(13, 21) = 1
P(18, 21) = 1
P(13, 22) = 1
P(19, 22) = 1
P(20, 23) = 1
P(21, 24) = 1
P(22, 25) = 1
P(23, 26) = 1
P(24, 26) = 1
P(25, 27) = 1
P(26, 28) = 1
P(24, 29) = 1
P(27, 29) = 1
P(28, 30) = 1
P(29, 30) = 1

'작업시간 tim

BM = 10000

t(1, 1) = 54
t(2, 1) = 144
t(3, 1) = 138
t(4, 1) = 144
t(5, 1) = 24
t(6, 1) = 102
t(7, 1) = 300
t(8, 1) = 114
t(9, 1) = 36
t(10, 1) = 114
t(11, 1) = 84
t(12, 1) = 72
t(13, 1) = 72
t(14, 1) = 24
t(15, 1) = 90
t(16, 1) = 72
t(17, 1) = 60
t(18, 1) = 360
t(19, 1) = 36
t(20, 1) = 114
t(21, 1) = 96
t(22, 1) = 24
t(23, 1) = 30
t(24, 1) = 78
t(25, 1) = 12
t(26, 1) = 270
t(27, 1) = 78
t(28, 1) = 90
t(29, 1) = 150
t(30, 1) = 36

t(1, 2) = BM
t(2, 2) = 70
t(3, 2) = BM
t(4, 2) = 86
t(5, 2) = BM
t(6, 2) = BM
t(7, 2) = BM
t(8, 2) = 58
t(9, 2) = 14
t(10, 2) = 70
t(11, 2) = BM
t(12, 2) = 44
t(13, 2) = 28
t(14, 2) = 12
t(15, 2) = BM
t(16, 2) = 42
t(17, 2) = BM
t(18, 2) = BM
t(19, 2) = BM
t(20, 2) = 64
t(21, 2) = BM
t(22, 2) = 14
t(23, 2) = BM
t(24, 2) = 48
t(25, 2) = BM
t(26, 2) = BM
t(27, 2) = 50
t(28, 2) = BM
t(29, 2) = BM
t(30, 2) = 22

t(1, 3) = 64
t(2, 3) = BM
t(3, 3) = 160
t(4, 3) = 142
t(5, 3) = BM
t(6, 3) = BM
t(7, 3) = 380
t(8, 3) = 108
t(9, 3) = BM
t(10, 3) = 110
t(11, 3) = BM
t(12, 3) = BM
t(13, 3) = 70
t(14, 3) = BM
t(15, 3) = BM
t(16, 3) = BM
t(17, 3) = 68
t(18, 3) = BM
t(19, 3) = 42
t(20, 3) = BM
t(21, 3) = 86
t(22, 3) = BM
t(23, 3) = 30
t(24, 3) = BM
t(25, 3) = 12
t(26, 3) = 290
t(27, 3) = 94
t(28, 3) = BM
t(29, 3) = 154
t(30, 3) = BM


maxJ = NJ

maxM(1) = 3
maxM(2) = 2
maxM(3) = 2

W(1) = 100000
W(2) = 100000
W(3) = 60000

CT = 250


    
Open App.Path & "\LB.ltx" For Output As #1 ' 출력을 위해 파일을 엽니다.

'목적함수

strX = ""

    For j = 1 To maxJ
    
        For k = 1 To NM
    
            strX = strX & " + " & W(k) & " N" & j & k
        Next

   Next
            
strX = "Min " & strX
    
Print #1, strX
strX = ""



'제약식

Print #1, "ST"
strX = ""


'1번 제약식

strX = ""

For i = 1 To NJ

    For j = 1 To maxJ
    
        For k = 1 To NM
    
            strX = strX & " + " & " X" & i & "_" & j & k
        Next

   Next
            
    strX = strX & " = 1 "
    
    Print #1, strX
    strX = ""

Next


'2번 제약식


strX = ""

For i = 1 To NJ

For ii = i + 1 To NJ
    
   
   If P(i, ii) = 1 Then
                
        For j = 1 To maxJ
    
            For k = 1 To NM
        
                strX = strX & " + " & j & " X" & i & "_" & j & k & " - " & j & " X" & ii & "_" & j & k
            
            Next
    
        Next
                
        strX = strX & " < 0"
        
        Print #1, strX
        strX = ""
    
     End If

    
Next
    
Next


'3번 제약식

strX = ""

For j = 1 To maxJ

    For k = 1 To NM


        For i = 1 To NJ
    
                  
            strX = strX & " + " & t(i, k) & " X" & i & "_" & j & k
                   
            
        Next
    
        strX = strX & " - " & CT & " N" & j & k
        
        strX = strX & " < 0"
    
        Print #1, strX
        strX = ""
    Next
    
Next

'strX = ""
'
'For j = 1 To maxJ
'
'    For i = 1 To NJ
'
'        For k = 1 To NM
'
'
'            strX = strX & " + " & t(i, k) & " X" & i & "_" & j & "_" & k
'
'
'        Next
'    Next
'
'        For k = 1 To NM
'
'          strX = strX & " - " & CT & " N" & j & "_" & k
'
'        Next
'
'    strX = strX & " < 0"
'
'    Print #1, strX
'    strX = ""
'
'Next



'4번 제약식

strX = ""


    For j = 1 To maxJ
    
        For k = 1 To NM
    
            strX = strX & " + " & " Y" & j & k
        Next
        
        strX = strX & " < 1 "
    
        Print #1, strX
        strX = ""

   Next
            




'5번 제약식

strX = ""


    For j = 1 To maxJ
    
        For k = 1 To NM
    
            strX = strX & " + " & " N" & j & k & " - " & "1000 Y" & j & k
            
            strX = strX & " < 0 "
    
            Print #1, strX
            strX = ""

            
        Next
        

   Next
   
            
'6번 제약식

strX = ""


    For j = 1 To maxJ - 1

        For k = 1 To NM

            strX = strX & " + " & "1000 N" & j & k & " - " & "N" & j + 1 & k
        Next
            
            strX = strX & " > 0 "

            Print #1, strX
            strX = ""

   Next

'    For j = 1 To maxJ - 1
'
'        For k = 1 To NM
'
'            strX = strX & " + " & "Y" & j & "_" & k & " - " & "Y" & j + 1 & "_" & k
'
'            strX = strX & " > 0 "
'
'            Print #1, strX
'            strX = ""
'
'
'        Next
'
'
'   Next


'7번 제약식
strX = ""

For j = 1 To maxJ
    For k = 1 To NM

            strX = "N" & j & k & " < " & maxM(k)
            
   
            Print #1, strX
            strX = ""
    Next
Next

'끝
Print #1, "END"

'정수선언

For i = 1 To NJ

    For j = 1 To maxJ
    
        For k = 1 To NM
    
                  
            strX = "INT X" & i & "_" & j & k
                   
            Print #1, strX
            
        Next
    Next

Next

    For j = 1 To maxJ
    
        For k = 1 To NM
    
                  
            strX = "INT Y" & j & k
                   
            Print #1, strX
            
        Next
    Next

    For j = 1 To maxJ
    
        For k = 1 To NM
    
                  
            strX = "GIN N" & j & k
                   
            Print #1, strX
            
        Next
    Next
    
    
    


Close #1

Text1.Text = "완료"

End Sub

