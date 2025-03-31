## Conspiracy theory

You've gotta believe me bro there's something here man. It's not a conspiracy. It's real im telling you!

## Solution
- we have a pdf which has 14 pages and in the last page i see some audio file type thing
![ss1](ss1.png)
- so i tried this for almost an hour and finally extracted that audio stream by
```bash 
stapat@stapat:~/ehax/CTF/tamu/cons$ pdfextract -o extracted conspiracy.pdf
Extracted 65 PDF streams to 'extracted/streams'.
Extracted 0 scripts to 'extracted/scripts'.
Extracted 0 attachments to 'extracted/attachments'.
Extracted 7 fonts to 'extracted/fonts'.
Extracted 34 images to 'extracted/images'.
stapat@stapat:~/ehax/CTF/tamu/cons$ file extracted/streams/*
extracted/streams/stream_193.dmp: data
extracted/streams/stream_200.dmp: ASCII text
extracted/streams/stream_203.dmp: data
extracted/streams/stream_205.dmp: data
extracted/streams/stream_213.dmp: ASCII text
extracted/streams/stream_219.dmp: data
extracted/streams/stream_282.dmp: data
extracted/streams/stream_293.dmp: ASCII text
extracted/streams/stream_2.dmp:   ASCII text
extracted/streams/stream_305.dmp: ASCII text
extracted/streams/stream_323.dmp: data
extracted/streams/stream_326.dmp: ASCII text
extracted/streams/stream_330.dmp: ASCII text
extracted/streams/stream_333.dmp: data
extracted/streams/stream_335.dmp: data
extracted/streams/stream_338.dmp: data
extracted/streams/stream_340.dmp: data
extracted/streams/stream_357.dmp: data
extracted/streams/stream_361.dmp: data
extracted/streams/stream_368.dmp: ASCII text
extracted/streams/stream_373.dmp: ASCII text
extracted/streams/stream_388.dmp: data
extracted/streams/stream_39.dmp:  ASCII text
extracted/streams/stream_405.dmp: data
extracted/streams/stream_406.dmp: data
extracted/streams/stream_411.dmp: ASCII text
extracted/streams/stream_421.dmp: data
extracted/streams/stream_42.dmp:  data
extracted/streams/stream_438.dmp: data
extracted/streams/stream_440.dmp: data
extracted/streams/stream_442.dmp: data
extracted/streams/stream_444.dmp: data
extracted/streams/stream_44.dmp:  data
extracted/streams/stream_451.dmp: ASCII text
extracted/streams/stream_454.dmp: data
extracted/streams/stream_458.dmp: data
extracted/streams/stream_483.dmp: data
extracted/streams/stream_493.dmp: data
extracted/streams/stream_495.dmp: data
extracted/streams/stream_502.dmp: ASCII text
extracted/streams/stream_507.dmp: data
extracted/streams/stream_509.dmp: data
extracted/streams/stream_511.dmp: MPEG ADTS, layer III, v2,  48 kbps, 24 kHz, Monaural
extracted/streams/stream_514.dmp: data
extracted/streams/stream_517.dmp: TrueType Font data, 12 tables, 1st "cmap", 58 names, Unicode, \251 2017 The Monotype Corporation. All Rights Reserved. 
extracted/streams/stream_520.dmp: ASCII text
extracted/streams/stream_522.dmp: TrueType Font data, 12 tables, 1st "cmap", 42 names, Unicode, \251 2018 Microsoft Corporation. All rights reserved.ConsolasRegularMicrosoft: ConsolasConsolasV
extracted/streams/stream_525.dmp: ASCII text
extracted/streams/stream_527.dmp: TrueType Font data, 12 tables, 1st "cmap", 58 names, Unicode, \251 2017 The Monotype Corporation. All Rights Reserved. 
extracted/streams/stream_530.dmp: ASCII text
extracted/streams/stream_532.dmp: TrueType Font data, 12 tables, 1st "cmap", 58 names, Unicode, \251 2017 The Monotype Corporation. All Rights Reserved. 
extracted/streams/stream_535.dmp: ASCII text
extracted/streams/stream_537.dmp: TrueType Font data, 12 tables, 1st "cmap", 47 names, Unicode, type 13 string, Microsoft supplied font. You may use this font to create, display, and print content as permitte
extracted/streams/stream_540.dmp: ASCII text
extracted/streams/stream_542.dmp: TrueType Font data, 12 tables, 1st "cmap", 45 names, Unicode, \251 2018 Microsoft Corporation. All Rights Reserved.
extracted/streams/stream_545.dmp: ASCII text
extracted/streams/stream_547.dmp: TrueType Font data, 12 tables, 1st "cmap", 47 names, Unicode, type 13 string, Microsoft supplied font. You may use this font to create, display, and print content as permitte
extracted/streams/stream_550.dmp: ASCII text
extracted/streams/stream_554.dmp: Microsoft color profile 2.1, type lcms, RGB/XYZ-mntr device by lcms, 8796 bytes, 25-3-2025 21:52:40 "sRGB built-in"
extracted/streams/stream_557.dmp: Unicode text, UTF-8 text
extracted/streams/stream_59.dmp:  ASCII text
extracted/streams/stream_64.dmp:  data
extracted/streams/stream_66.dmp:  data
extracted/streams/stream_68.dmp:  data
extracted/streams/stream_71.dmp:  data
```
- we see stream_511.dmp is a MPEG ADTS, layer III, v2,  48 kbps, 24 kHz, Monaural
```bash
stapat@stapat:~/ehax/CTF/tamu/cons$ ffprobe extracted/streams/stream_511.dmp 
ffprobe version 6.1.1-3ubuntu5 Copyright (c) 2007-2023 the FFmpeg developers
  built with gcc 13 (Ubuntu 13.2.0-23ubuntu3)
  configuration: --prefix=/usr --extra-version=3ubuntu5 --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --disable-omx --enable-gnutls --enable-libaom --enable-libass --enable-libbs2b --enable-libcaca --enable-libcdio --enable-libcodec2 --enable-libdav1d --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libglslang --enable-libgme --enable-libgsm --enable-libharfbuzz --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzimg --enable-openal --enable-opencl --enable-opengl --disable-sndio --enable-libvpl --disable-libmfx --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r --enable-ladspa --enable-libbluray --enable-libjack --enable-libpulse --enable-librabbitmq --enable-librist --enable-libsrt --enable-libssh --enable-libsvtav1 --enable-libx264 --enable-libzmq --enable-libzvbi --enable-lv2 --enable-sdl2 --enable-libplacebo --enable-librav1e --enable-pocketsphinx --enable-librsvg --enable-libjxl --enable-shared
  libavutil      58. 29.100 / 58. 29.100
  libavcodec     60. 31.102 / 60. 31.102
  libavformat    60. 16.100 / 60. 16.100
  libavdevice    60.  3.100 / 60.  3.100
  libavfilter     9. 12.100 /  9. 12.100
  libswscale      7.  5.100 /  7.  5.100
  libswresample   4. 12.100 /  4. 12.100
  libpostproc    57.  3.100 / 57.  3.100
extracted/streams/stream_511.dmp: Invalid data found when processing input
stapat@stapat:~/ehax/CTF/tamu/cons$ xxd extracted/streams/stream_511.dmp | head -n 100
00000000: fff3 64c4 0000 0003 4800 0000 004c 414d  ..d.....H....LAM
00000010: 4555 5555 4c41 4d45 332e 3130 3055 5555  EUUULAME3.100UUU
00000020: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000030: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000040: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000050: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000060: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000070: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000080: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000090: 0b78 4df8 7c00 0003 4800 0000 0055 5555  .xM.|...H....UUU
000000a0: 5555 5555 4c41 4d45 332e 3130 3055 5555  UUUULAME3.100UUU
000000b0: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
000000c0: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
000000d0: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
000000e0: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
000000f0: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000100: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000110: 5555 5555 5555 5555 5555 5555 5555 5555  UUUUUUUUUUUUUUUU
00000120: eb04 25ac 7c00 0003 4800 0000 0055 5555  ..%.|...H....UUU
00000130: 5555 5555 f289 16f1 4a0c cf87 e402 a7e0  UUUU....J.......
00000140: 3241 4606 c73f c381 0c84 0150 1ccf fc42  2AF..?.....P...B
00000150: 414a 0694 1f28 071f ffc0 5761 f500 a580  AJ...(....Wa....
00000160: 1ac0 a000 903f ff00 e209 6874 e037 30da  .....?....ht.70.
00000170: 0038 6089 7fff f8a0 4050 616c ca45 4162  .8`.....@Pal.EAb
00000180: 1610 dbc1 61ff ffff 8854 050e 048e 4809  ....a....T....H.
00000190: dcbc 4345 c630 8634 2eac 38ff ffff fff2  ..CE.0.4..8.....
000001a0: a8cd 9788 d14b 8f64 c8b8 cb64 68b8 c933  .....K.d...dh..3
000001b0: e51a 5963 7c00 0003 4801 4000 0012 0e91  ..Yc|...H.@.....
000001c0: 67ff ffff ffff c89b 9648 9a47 489a 4749  g........H.GH.GI
000001d0: f48e 9133 6288 e394 c9f3 4299 3e49 b923  ...3b.....B.>I.#
000001e0: 3198 8526 9158 a446 9b60 1dfd db95 60df  1..&.X.F.`....`.
000001f0: 721c 9069 1ee7 69bd 5374 39a6 7b48 7255  r..i..i.St9.{HrU
00000200: 0472 8911 236f d32b 6b76 92fd 93bf e02a  .r..#o.+kv.....*
00000210: 4b5d f698 9817 650f 8380 d470 6ee5 9b6e  K]....e....pn..n
00000220: 9567 f91a 620f eda6 c6c1 9f51 190e 4e05  .g..b......Q..N.
00000230: 5044 2481 051d 7c38 b335 e9e9 989c 4a15  PD$...|8.5....J.
00000240: 8f20 58af ff26 fc39 e801 9098 000f 651f  . X..&.9......e.
00000250: 1aa3 4e05 0018 25da fd41 10da ea2e 397c  ..N...%..A....9|
00000260: daf4 8233 2da3 a19a fe7b 7389 2da3 1093  ...3-....{s.-...
00000270: e0d0 269b 2a7a b55f 55e5 714c 3ea5 bc9d  ..&.*z._U.qL>...
00000280: 080a cc5e 9b75 28a1 886a 04ee 72b9 fc26  ...^.u(..j..r..&
00000290: eb3d f4f7 5fda d639 3949 2f94 45b4 feda  .=.._..99I/.E...
000002a0: a58c d8b1 666e d4d7 3bce eb1e dfdf 6fe1  ....fn..;.....o.
000002b0: 85bf ef79 53b8 72fd ec32 c2e5 5bff fff0  ...yS.r..2..[...
000002c0: dc6a acfd 6b71 ab14 162a d4ff ffff fe5e  .j..kq...*.....^
000002d0: abe7 c799 e639 7322 865f 98c0 0252 9269  .....9s"._...R.i
000002e0: 1114 ddff 0fd7 29e1 a62f 82d6 f26c 6174  ......)../...lat
000002f0: a31f 6872 754e 4c1e 9bc5 f591 bc7f 8975  ..hruNL........u
00000300: 2a29 38e2 fde4 26b2 7ec8 903c 1088 6d89  *)8...&.~..<..m.
00000310: f616 e4fa 5d5b 67af 1a94 0cc8 f5a5 5b12  ....][g.......[.
00000320: 10e4 a06e 7bb7 f4cc 0a52 3bfb c396 043f  ...n{....R;....?
00000330: 879d 9173 2e58 23dd 924b bca3 d553 8b9e  ...s.X#..K...S..
00000340: 2919 e2db 5cec 36f7 d52c f337 f0aa c71d  )...\.6..,.7....
00000350: bddc 917d ef12 5bdf 6cef 25ff 1131 11b3  ...}..[.l.%..1..
00000360: 63db d189 8335 ac32 9a2f cf78 022d 5335  c....5.2./.x.-S5
00000370: b3a3 f6fe ca08 3037 8bb1 c379 1eb8 4322  ......07...y..C"
00000380: 436c 7509 c149 0cbe 2c33 bc87 19e4 af1c  Clu..I..,3......
00000390: e038 fabb 6d83 1d82 27ff ffff ffff ffff  .8..m...'.......
000003a0: ffff ffc9 2fff ffff ff14 dfff ffff f41b  ..../...........
000003b0: e927 7009 5c2a 9208 be99 beff c53a 60ef  .'p.\*.......:`.
000003c0: bd8c 0000 4a56 b35a 5abe 8d2a 01c2 483b  ....JV.ZZ..*..H;
000003d0: 9c72 1c89 c8fb 36dc 19a4 00fe cbe4 908d  .r....6.........
000003e0: cd52 bbff a9d7 2ed5 8963 ff23 6672 2587  .R.......c.#fr%.
000003f0: 9cd9 f58f 2f28 8af6 9de6 ca4d 48a9 14ab  ..../(.....MH...
00000400: 72ab 5e70 5c77 788c 3619 1b1f 1309 cc92  r.^p\wx.6.......
00000410: 15a1 5922 4422 0253 8450 3a4a a878 6553  ..Y"D".S.P:J.xeS
00000420: 0b27 1ed8 8b43 3ce6 5b7a f25a 1526 c4a4  .'...C<.[z.Z.&..
00000430: 0cb0 380f b8df d513 cf5c f833 4b7c c40f  ..8......\.3K|..
00000440: ae12 bbca d83f 59fc eb91 a2fe 6522 83f6  .....?Y.....e"..
00000450: 3e99 913e 3178 e9f3 e84d 20f1 f7ae 7c6a  >..>1x...M ...|j
00000460: 2bd2 91ff fa7f 6fcb 3cb4 d1f1 6973 4224  +.....o.<...isB$
00000470: aab6 ea64 4442 c521 72c6 e4da 6519 ce61  ...dDB.!r...e..a
00000480: be97 f98a 0f20 b926 d256 de4c 7ada 1181  ..... .&.V.Lz...
00000490: 865e a74a 90d6 56c5 b82d f17c b299 48b1  .^.J..V..-.|..H.
000004a0: a080 4cad 87b8 bcc3 640d 6e69 065f 48ab  ..L.....d.ni._H.
000004b0: 1931 c371 a355 e95f 166f 3563 e3cf a369  .1.q.U._.o5c...i
000004c0: dfef eeda 14ee 2490 a491 f0ed bbef bfce  ......$.........
000004d0: 0e1c 951a feb8 c111 31a0 2ffe 9b10 9502  ........1./.....
000004e0: 6283 42c2 ce08 8a8e 6080 68b0 c59e 1fa3  b.B.....`.h.....
000004f0: ffff ff7d 92ea 381d 2844 aa44 ce68 abc5  ...}..8.(D.D.h..
00000500: ddb0 ce79 49bd b837 f6ae 52d6 46b8 3ff7  ...yI..7..R.F.?.
00000510: 2a4d d168 0f20 52b2 de36 ca07 2e1f 1494  *M.h. R..6......
00000520: 2150 31ec bfee 8a02 1d64 0b7f 1b82 3299  !P1......d....2.
00000530: 24b6 d7d5 7b19 3f7b bbaa 857a 55ff 1607  $...{.?{...zU...
00000540: 6ffe 1e3c 90f0 8aae 2d3e 4930 bb6c 1a86  o..<....->I0.l..
00000550: 0472 efe5 fea8 6088 9ac3 fda9 9ea2 67f5  .r....`.......g.
00000560: 61fc 32ff ff84 fe4f cacd 5772 e093 7362  a.2....O..Wr..sb
00000570: 7a10 de21 a922 e64c 8db3 2982 a0a4 4a00  z..!.".L..)...J.
00000580: 07ff ffff f959 16d6 41b5 0524 8aa5 0495  .....Y..A..$....
00000590: 8812 0074 95bb c2be a3ec 0116 70ee f6f0  ...t........p...
000005a0: 2a30 aa8b 101e 8a3e c5f6 c187 2e83 98fc  *0.....>........
000005b0: 6b77 1b09 77e7 f2ee 3052 bb75 37aa 9126  kw..w...0R.u7..&
000005c0: 94b5 e7f3 c169 1735 2f22 98d5 0904 efad  .....i.5/"......
000005d0: 5b1f b4cd 67fe 5afb 51d7 f704 848a cfe6  [...g.Z.Q.......
000005e0: 5926 c753 cbdf f584 760a d4fa 5ce1 ffdf  Y&.S....v...\...
000005f0: 865a 9291 92eb 1edf a0d6 5944 c500 287a  .Z........YD..(z
00000600: ce0c 0c87 dcd6 6c47 f49f dc72 a7fc 3cda  ......lG...r..<.
00000610: 34d2 54b5 8bc5 f52a d000 0890 a34e 542b  4.T....*.....NT+
00000620: b10e 0196 6e8e 32d4 0ba0 98bd 34a9 11d0  ....n.2.....4...
00000630: e952 7d9b 1822 7c36 bd9e 8bca 5e16 1a2d  .R}.."|6....^..-
```
- we see that only the first header is correct else all are corrupted , so adding header 0xFFF364C4 after every 144 bytes as this is the length of the audio frame header in our case (calculated it or googled it(hehe))
```python
def fix_repeating_pattern(filename, pattern=0xfff364c4, interval=144):
    with open(filename, 'rb') as f:
        data = bytearray(f.read())
    
    pattern_bytes = pattern.to_bytes(4, 'big')
    
    for i in range(0, len(data), interval):
        if data[i:i+4] != pattern_bytes:
            print(f"Fixing mismatch at offset {i}: {data[i:i+4].hex()} -> {pattern_bytes.hex()}")
            data[i:i+4] = pattern_bytes
    
    with open(filename, 'wb') as f:
        f.write(data)
    
    print("done")
filename = "extracted/streams/stream_511.dmp"
fix_repeating_pattern(filename)
```
```bash
mpg321 extracted/streams/stream_511.dmp
High Performance MPEG 1.0/2.0/2.5 Audio Player for Layer 1, 2, and 3.
Version 0.3.2-1 (2012/03/25). Written and copyrights by Joe Drew,
now maintained by Nanakos Chrysostomos and others.
Uses code from various people. See 'README' for more!
THIS SOFTWARE COMES WITH ABSOLUTELY NO WARRANTY! USE AT YOUR OWN RISK!

Directory: extracted/streams
Playing MPEG stream from stream_511.dmp ...
MPEG 2.0 layer III, 48 kbit/s, 24000 Hz mono
```
```
flag = gigem{mp3_is_so_free_eletric_bageelee} ( something like this)
```
