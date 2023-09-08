import {Command, ux} from '@oclif/core'
import {exec} from 'node:child_process'

export default class Start extends Command {
  static description = 'Stop the Grepit Server'

  static examples = [`
    $ oex stop
  `]

  async run(): Promise<void> {
    ux.action.start('Stopping Server')
    exec('ps -ax | grep ../../proxy/server.js', (error, stdout, stderr) => {
      const matches = stdout.match(/(\S+)/) || []
      if (stderr) {
        throw new Error(stderr)
      }

      if (matches.length > 0 && matches[0]) {
        process.kill(Number.parseInt(matches[0], 10))
        ux.action.stop('Server Stopped')
        return
      }

      ux.action.stop('Server Not Running')
    })
  }
}
