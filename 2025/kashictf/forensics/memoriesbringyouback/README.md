# Memories Bring You back
 desription -A collection of images, a digital time capsule—preserved in this file. But is every picture really just a picture? A photographer once said, “Every image tells a story, but some stories are meant to stay hidden.” Maybe it’s time to inspect the unseen and find what’s been left behind.

---
# Solution
it was a 1000mb file which was a
 ```bash
stapat@stapat:~/ehax/CTF/kashi$ file chall
chall: DOS/MBR boot sector MS-MBR Windows 7 english at offset 0x163 "Invalid partition table" at offset 0x17b "Error loading operating system" at offset 0x19a "Missing operating system", disk signature 0x5032578b; partition 1 : ID=0x7, start-CHS (0x0,2,3), end-CHS (0x7e,254,63), startsector 128, 2041856 sectors
```
after mounting this i saw there were 500 images and some folders , tried digging in it but got  nothing useful so i randomly ran strings on the orignal file and got the flag
```bash
stapat@stapat:~/ehax/CTF/kashi$ strings chall | grep Kashi
KashiCTF{Fake_Flag}
KashiCTF{Fake_Flag}
KashiCTF{Fake_Flag}
KashiCTF{DF1R_g03555_Brrrr}
stapat@stapat:~/ehax/CTF/kashi$
```
got the  ```KashiCTF{DF1R_g03555_Brrrr}```