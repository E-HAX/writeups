# Lost Frequencies 

## Challenge Description:
The challenge provides the following hint:
```
Zeroes, ones, dots and dashes
Data streams in bright flashes
```
And the given data:
```
111 0000 10 111 1000 00 10 01 010 1011 11 111 010 000 0
```

## Step 1: Understanding the Hint
The phrase **"Zeroes, ones, dots and dashes"** suggests that the data might be related to **binary and Morse code**. This gives us a direction to investigate.

## Step 2: Binary Interpretation
The given sequence consists of binary numbers. First, we merge them into a single binary string:
```
1110000101111000001001010101011111110100000
```
Then, we try converting this binary data into ASCII characters:
```
'áx%_Ð'
```
This output seems random and not meaningful, suggesting that ASCII decoding is **not the intended approach**.

## Step 3: Morse Code Analysis
Since the hint also mentions **dots and dashes**, we map binary digits to Morse:
- `1` → `-` (Dash)
- `0` → `.` (Dot)

Converting each binary group:
```
--- .... -. --- -... .. -. .- .-. -.-- -- --- .-. ... .
```

Using Morse code translation:
```
O H N O B I N A R Y M O R S E
```
This translates to **"OHNOBINARYMORSE"**.

## Step 4: Interpreting the Flag
The phrase **"Oh No Binary Morse"** suggests that the challenge was a trick—perhaps leading solvers to focus on binary while the actual solution was Morse code.

The final flag format might be:
```
KashiCTF{OHNOBINARYMORSE}
```

## Conclusion
This challenge cleverly combined **binary representation and Morse code** to mislead solvers into thinking it was a binary decoding task, when in reality, the binary was just a way to encode Morse. By carefully analyzing the hint and different decoding methods, we successfully extracted the correct flag!
