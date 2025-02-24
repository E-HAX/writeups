# Solution
1. we got a image named pasta.jpg , so like a normal jpg i did everything which someone should do , and checking for some data appened in the hex of the jpg i found something  unusual 
```bash
stapat@stapat:~/ehax/CTF/kashi$ xxd pasta.jpg | tail
0000b6f0: 2eb7 8c66 31a8 c88a aeb7 8c66 31a8 c88a  ...f1......f1...
0000b700: 5d6f 18cc 6351 9114 5bc6 3519 1911 45bc  ]o..cQ..[.5...E.
0000b710: 6331 8d46 4452 eb78 c663 1a8c 88ae eb78  c1.FDR.x.c.....x
0000b720: c663 1a8c 88a5 d6f1 8d46 4644 54b7 8c6a  .c.......FFDT..j
0000b730: 3232 228b 78c6 a323 2228 b232 3232 228b  22".x..#"(.222".
0000b740: 2323 2322 28b2 378c 6a32 228b 2323 2322  ###"(.7.j2".###"
0000b750: 28bf ffd9 baab aaab bbaa baab abba baba  (...............
0000b760: aaab aaba aaaa abaa baaa aaab aaaa aaaa  ................
0000b770: baba abab aaba baab abab abba aaab aabb  ................
0000b780: abab baba baab abaa aabb aaaa bba0       ..............
```
2. it was some kind of cipher so after reading the challenge text and analyizng the image that the pasta shown in the image was tomato bacon pasta 
3. so i searched online for ciphers like tomato cipher , pasta cipher and i found something like bacon cipher 
4. decoding the text from bacon cipher gave me the flag
5. flag - ```KashiCTF{theywerereallllycooking}```