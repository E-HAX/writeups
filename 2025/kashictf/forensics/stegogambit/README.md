# Solution
1. in this challenge we got a image of a chess game and it seemed a normal chess game not any variant 
2. after reading the challenge description i analyzed the position and got that only one checkmate is possible and this can also be analyzed by some engines online (ex - [lichess](https://lichess.org/editor), stockfish)
```
Bh1 , Kxa2 , Qg2#https://lichess.org/editor
```
3. as this was a forensics challenge not a chess challenge (lol) so i tried some basic tools , did exiftool and got ```Use the moves as a key to the flag, separated by _```
4. the ``key`` word should be the passphrase for steghide so i tried this ```Bh1_Kxa2_Qg2#``` but this didnt work so i tried different permutations and combinations for next few hours 
5. finally i tried the passphrase as ```Bh1Kxa2_Qg2#``` , it had a file **flag.txt** and it had the flag

```bash
stapat@stapat:~/ehax/CTF/kashi$ xdg-open chall.jpg 
stapat@stapat:~/ehax/CTF/kashi$ exiftool chall.jpg 
ExifTool Version Number         : 12.76
File Name                       : chall.jpg
Directory                       : .
File Size                       : 49 kB
File Modification Date/Time     : 2025:02:23 00:05:25+05:30
File Access Date/Time           : 2025:02:25 02:00:48+05:30
File Inode Change Date/Time     : 2025:02:23 00:05:42+05:30
File Permissions                : -rw-rw-r--
File Type                       : JPEG
File Type Extension             : jpg
MIME Type                       : image/jpeg
JFIF Version                    : 1.01
Resolution Unit                 : None
X Resolution                    : 1
Y Resolution                    : 1
Comment                         : Use the moves as a key to the flag, separated by _
Image Width                     : 817
Image Height                    : 815
Encoding Process                : Baseline DCT, Huffman coding
Bits Per Sample                 : 8
Color Components                : 3
Y Cb Cr Sub Sampling            : YCbCr4:2:0 (2 2)
Image Size                      : 817x815
Megapixels                      : 0.666
stapat@stapat:~/ehax/CTF/kashi$ xdg-open chall.jpg 
stapat@stapat:~/ehax/CTF/kashi$ exiftool chall.jpg 
ExifTool Version Number         : 12.76
File Name                       : chall.jpg
Directory                       : .
File Size                       : 49 kB
File Modification Date/Time     : 2025:02:23 00:05:25+05:30
File Access Date/Time           : 2025:02:25 02:00:48+05:30
File Inode Change Date/Time     : 2025:02:23 00:05:42+05:30
File Permissions                : -rw-rw-r--
File Type                       : JPEG
File Type Extension             : jpg
MIME Type                       : image/jpeg
JFIF Version                    : 1.01
Resolution Unit                 : None
X Resolution                    : 1
Y Resolution                    : 1
Comment                         : Use the moves as a key to the flag, separated by _
Image Width                     : 817
Image Height                    : 815
Encoding Process                : Baseline DCT, Huffman coding
Bits Per Sample                 : 8
Color Components                : 3
Y Cb Cr Sub Sampling            : YCbCr4:2:0 (2 2)
Image Size                      : 817x815
Megapixels                      : 0.666
stapat@stapat:~/ehax/CTF/kashi$ steghide extract -sf chall.jpg -p Bh1Kxa2_Qg2#
the file "flag.txt" does already exist. overwrite ? (y/n) y
wrote extracted data to "flag.txt".
stapat@stapat:~/ehax/CTF/kashi$ 
stapat@stapat:~/ehax/CTF/kashi$ cat flag.txt 
KashiCTF{573g0_g4m617_4cc3p73d}
```