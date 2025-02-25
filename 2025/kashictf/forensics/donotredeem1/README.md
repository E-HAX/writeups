# do not redeem #1 - writeup

* navigate to `/Downloads/kitler's-phone (2)/data/data/com.android.providers.telephony/databases` 
* open mmssmm.db using sqllite3
* sqlite> select * from sms;
	1|1|57575022||1740251608654|1740251606000|0|1|-1|1|0||839216 is your Amazon OTP. Don't share it with anyone.||0|1|0|com.google.android.apps.messaging|1
	2|2|AX-AMZNIN||1740251865569|1740251864000|0|1|-1|1|0||Order placed with order id: PO3663460903896.

	Thank you for choosing Amazon as your shopping destination.

	Remaining Amazon Pay balance: INR0.69||0|1|0|com.google.android.apps.messaging|1

* got the flag