# Self Destruct Writeup

- First botted up the VM, but it closed after sometime so couldnt do anything with it.
- Run strings on the .vdi file and saved the output in a separate text file.
- Searched for `KashiCTF{` in the file and found the part1 of the flag.
- This gave away the format in which the parts were stored, so the next time onwards i searched for `fLaG Part x:`

```
Flag: KashiCTF{rm_rf_no_preserve_root_Am_1_Right??_No_Err0rs_4ll0wed_Th0}