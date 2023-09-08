import {Flags, Command, ux} from '@oclif/core'
import {checkApiKey, addApiKey, addApiUrl} from '../../proxy/utils'
import {spawn} from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'

export default class Start extends Command {
  static description = 'Start the GrepIt Server'

  static examples = [`
    $ oex start
  `]

  static flags = {
    port: Flags.string({char: 'p'}),
  }

  async run(): Promise<void> {
    const {flags} = await this.parse(Start)
    ux.action.start('Starting Server')

    try {
      fs.accessSync(`${this.config.home}/.wakatime`)
    } catch {
      throw new Error('Please install a wakatime plugin')
    }

    let apiKey = checkApiKey()
    if (!apiKey) {
      apiKey = await ux.prompt('Enter your API key')
      await addApiKey(apiKey)
    }

    try {
      fs.accessSync(this.config.dataDir)
    } catch {
      await fs.mkdirSync(this.config.dataDir, {recursive: true})
    }

    let user = ''
    try {
      const userConfig = await fs.readFileSync(path.join(this.config.configDir, 'config.json'))
      const {username} = JSON.parse(userConfig.toString())
      user = username
    } catch {
      const username = await ux.prompt('Enter your Username')
      await fs.mkdirSync(this.config.configDir, {recursive: true})
      fs.writeFileSync(path.join(this.config.configDir, 'config.json'), JSON.stringify({username}))
      user = username
    }

    addApiUrl()
    const port = flags.port || '8888'
    const cmd = process.env.NODE_ENV === 'development' ? 'node' : `${__dirname}/../../../bin/node`
    const child = spawn(cmd, [`${__dirname}/../../proxy/server.js`, user, this.config.dataDir, port], {
      detached: true,
      stdio: 'ignore',
    })

    child.unref()

    ux.action.stop(`Server Started at http://localhost:${port}/`)
  }
}
