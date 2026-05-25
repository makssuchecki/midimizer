import pretty_midi


def events_to_midi(events, output_path="output.mid", tempo=120):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    current_time = 0.0
    time_scale = 0.01

    active_notes = {}

    for event in events:
        if event["type"] == "time_shift":
            current_time += event["value"] * time_scale

        elif event["type"] == "note_on":
            pitch = event["pitch"]
            velocity = event["velocity"]

            active_notes[pitch] = {
                "start": current_time,
                "velocity": velocity
            }

        elif event["type"] == "note_off":
            pitch = event["pitch"]

            if pitch in active_notes:
                note = pretty_midi.Note(
                    velocity=active_notes[pitch]["velocity"],
                    pitch=pitch,
                    start=active_notes[pitch]["start"],
                    end=current_time
                )
                instrument.notes.append(note)
                del active_notes[pitch]

    midi.instruments.append(instrument)
    midi.write(output_path)

    print(f"Saved MIDI to {output_path}")