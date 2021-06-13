from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

list = ['statue of liberty', 'mount rushmore', 'atomium', 'Monument to Party Founding', 'chichen itza mexico', 'Abu Simbel', 'Three Kings Monument']
# list = ['Eiffel Tower','Empire State Building','Colosseum','Tower of Pisa','Taj Mahal','Sydney Opera House','Marina Bay Sands','Parthenon','Willis Tower','Chrysler Building','Burj Khalifa','Sagrada Familia','Hagia Sophia','Big Ben','The Shard','Solomon R. Guggenheim Museum','Roanoke Virginia National Museum', 'Flatiron Building','Palace of Versailles','Pantheon','The White House','Angkor Wat','Burj Al Arab Hotel','One World Trade Center','St. Pauls Cathedral','Space Needle','Falling Water','CN Tower','Arc de Triomphe','Taipei','Potala Palace', 'Lhasa','The Pentagon (Arlington)','Forbidden City', 'Beijing','The Gherkin','Buckingham Palace','Casa Mila','Gateway Arch','Erechtheion']

for i in list:
    response().download(i, 10)


