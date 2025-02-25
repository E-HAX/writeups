# SuperFastApi
Made my verty first API!

However I have to still integrate it with a frontend so can't do much at this point lol.


## Solution
- run go buster on given api
- will get a route `/docs`
- on `/docs` various end points are there to test
    - / (root) - GET
    - /get/{username} - GET
    - /create/{username} - POST
    - /update/{username} - PUT
    - /flag/{username} - GET
- first created a user with username `admin` using `/create/{username}` endpoint with following data -
  request body -

  ```
  {
  "fname": "john",
  "lname": "cena",
  "email": "john@cena.com",
  "gender": "male"
  }
  ```
  
  Got message "User created!"
  
- check for flag at `/flag/{username}`, you will get -
  ```
  {
  "error": "Only for admin"
  }
  ```

- Now update user with following request body using `/update/{username}` route -
  ```
  {
  "fname": "john",
  "lname": "cena",
  "email": "john@cena.com",
  "gender": "male",
  "admin": true,
  "role": "admin"
  }
  ```

  Will get message  "User created!" again after user data is updated

- check again for flag, and you'll get the flag
