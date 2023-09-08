/* eslint-disable unicorn/prefer-module */
const fs = require('node:fs')
const {homedir} = require('node:os')

const filePath = `${homedir()}/.wakatime.cfg`
const apiUrlString = 'api_url = http://localhost:8888'

const removeApiUrl = () => {
  const data = fs.readFileSync(`${homedir()}/.wakatime.cfg`, 'utf8')
  if (data.includes(apiUrlString)) {
    const newFile = data.replace(apiUrlString, '')
    if (newFile.length > 0) {
      fs.writeFileSync(filePath, newFile, 'utf8')
    }
  }
}

module.exports = {
  removeApiUrl,
}
