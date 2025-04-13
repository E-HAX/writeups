## Beginner web
```
Flag part 1: w3b_
Flag Part 2: br0w53r5_4r3_
Flag Part 3: c0mpl1c473d
```

```javascript
const CryptoJS = require("crypto-js");

let n = "flagPart2_3";  // Decryption key
let r = "U2FsdGVkX1/oCOrv2BF34XQbx7f34cYJ8aA71tr8cl8="; // Encrypted flag part 2
let o = "U2FsdGVkX197aFEtB5VUIBcswkWs4GiFPal6425rsTU="; // Encrypted flag part 3

// Decrypt flag parts
let flagPart2 = CryptoJS.AES.decrypt(r, n).toString(CryptoJS.enc.Utf8);
let flagPart3 = CryptoJS.AES.decrypt(o, n).toString(CryptoJS.enc.Utf8);

console.log("Flag Part 2:", flagPart2);
console.log("Flag Part 3:", flagPart3);

```
