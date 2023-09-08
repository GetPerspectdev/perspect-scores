import * as fs from 'node:fs'
import {homedir} from 'node:os'

const filePath = `${homedir()}/.wakatime.cfg`
const apiUrlString = 'api_url = http://localhost:8888'
const apiKeyString = 'api_key ='

export interface Configs {
  username: string,
  sync?: boolean,
  remoteUrl?: string,
  port?: number
}

export const checkApiKey = (): string => {
  const apiKeyRegEx = /api_key\s*=\s*(.*)/
  try {
    const data = fs.readFileSync(filePath, 'utf8')
    const matches = data.match(apiKeyRegEx)
    return matches ? matches[1] : ''
  } catch {
    throw new Error('Please install a wakatime plugin before using grepit')
  }
}

export const addApiUrl = (): void => {
  const data = fs.readFileSync(filePath, 'utf8')
  let newFile = data
  if (data.includes(apiUrlString)) {
    console.log('String already exists in the file.')
    return
  }

  const lastCharacter = data.slice(-1)
  if (lastCharacter !== '\n') {
    newFile += '\n'
  }

  newFile += apiUrlString

  fs.writeFileSync(filePath, newFile, 'utf8')
}

export const removeApiUrl = (): void => {
  const data = fs.readFileSync(`${homedir()}/.wakatime.cfg`, 'utf8')
  if (data.includes(apiUrlString)) {
    const newFile = data.replace(apiUrlString, '')
    if (newFile.length > 0) {
      fs.writeFileSync(filePath, newFile, 'utf8')
    }
  }
}

export const addApiKey = (apiKey: string): void => {
  const data = fs.readFileSync(filePath, 'utf8')
  let newFile = data
  if (data.includes(apiKeyString)) {
    console.log('String already exists in the file.')
    return
  }

  const lastCharacter = data.slice(-1)
  if (lastCharacter !== '\n') {
    newFile += '\n'
  }

  newFile += `${apiKeyString} ${apiKey}`

  fs.writeFileSync(filePath, newFile, 'utf8')
}
