use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Text};
use ratatui::widgets::{Block, Borders, Paragraph};

use crate::backend::describe_backend;
use crate::config::AppConfig;
use crate::types::{BackendKind, MODEL_HF_REPO};

pub fn run_tui(config: AppConfig) -> Result<()> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = tui_loop(&mut terminal, config);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

fn tui_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    config: AppConfig,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            let area = frame.area();
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(4),
                    Constraint::Min(12),
                    Constraint::Length(3),
                ])
                .split(area);

            let header = Paragraph::new(Text::from(vec![
                Line::from("fpga-whisper"),
                Line::from("Hybrid Whisper runtime skeleton for Rust + CTranslate2 + FPGA."),
            ]))
            .block(Block::default().title("Overview").borders(Borders::ALL))
            .style(Style::default().add_modifier(Modifier::BOLD));
            frame.render_widget(header, chunks[0]);

            let ct2 = describe_backend(BackendKind::Ct2Python);
            let fpga_sim = describe_backend(BackendKind::FpgaSim);
            let fpga = describe_backend(BackendKind::FpgaHybrid);
            let body = Paragraph::new(Text::from(vec![
                Line::from(format!("Baked-in model: {MODEL_HF_REPO}")),
                Line::from(format!(
                    "Worker launcher: {} {}",
                    config.worker_launcher.display(),
                    config.worker_launcher_args.join(" ")
                )),
                Line::from(format!("Worker script: {}", config.worker_script.display())),
                Line::from(""),
                Line::from("Backends"),
                Line::from(format!("- {}: {}", ct2.id.display_name(), ct2.summary)),
                Line::from(format!(
                    "- {}: {}",
                    fpga_sim.id.display_name(),
                    fpga_sim.summary
                )),
                Line::from(format!("- {}: {}", fpga.id.display_name(), fpga.summary)),
                Line::from(""),
                Line::from("Target FPGA ownership"),
                Line::from("- feature extraction"),
                Line::from("- encoder"),
                Line::from("- decoder math"),
                Line::from(""),
                Line::from("Next milestone"),
                Line::from("- add context carry-over and timestamps to the CTranslate2 baseline"),
                Line::from("- replace the simulator bridge with iverilog/vvp execution"),
            ]))
            .block(Block::default().title("Project Map").borders(Borders::ALL));
            frame.render_widget(body, chunks[1]);

            let footer = Paragraph::new("Press q to quit.")
                .block(Block::default().title("Controls").borders(Borders::ALL));
            frame.render_widget(footer, chunks[2]);
        })?;

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    break;
                }
            }
        }
    }

    Ok(())
}
